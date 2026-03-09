"""Standalone embedding generation script.

Runs the configured encoder(s) over the train and test splits of a dataset
**once**, then writes the resulting :class:`~deepref.embedding.vector_database.VectorDatabase`
files to an output directory.  Saved databases can be loaded later and fed
directly to :class:`~deepref.experiments.run_combine_embeddings_experiments.VectorDBRETrainer`,
bypassing the (slow, GPU-bound) encoder at every training epoch.

Typical workflow
----------------
1. **Generate** — run this script to populate an output directory::

       python deepref/experiments/generate_embeddings.py \\
           output_dir=embeddings/semeval2010

2. **Train** — pass ``vector_db.enabled=true`` and ``vector_db.save_dir`` to
   the experiment runner, or load the databases manually::

       from deepref.embedding.vector_database import VectorDatabase
       train_vdb = VectorDatabase.load("embeddings/semeval2010/semeval2010_relation_bow_sdp_train")
       test_vdb  = VectorDatabase.load("embeddings/semeval2010/semeval2010_relation_bow_sdp_test")

Usage
-----
Default config (relation + bow_sdp encoder, semeval2010 dataset)::

    python deepref/experiments/generate_embeddings.py

Custom output directory::

    python deepref/experiments/generate_embeddings.py output_dir=embeddings/semeval2010

Different encoder or dataset::

    python deepref/experiments/generate_embeddings.py \\
        encoder1=llm encoder2=verbalized_sdp dataset=ddi

With preprocessing pipeline (StandardScaler → PCA(0.95) → L2 Norm)::

    python deepref/experiments/generate_embeddings.py fit_pipeline=true

All combinations via multirun::

    python deepref/experiments/generate_embeddings.py --multirun \\
        dataset=semeval2010,ddi \\
        encoder1=relation,llm \\
        encoder2=bow_sdp,verbalized_sdp

Config keys
-----------
output_dir   : destination directory (relative to the original working directory).
batch_size   : encoder forward-pass batch size (default 64).
device       : ``"cuda"`` or ``"cpu"``.
fit_pipeline : if ``true``, fit StandardScaler → PCA(0.95) → L2 Norm on the
               train VDB and apply the fitted pipeline to the test VDB.
seed         : random seed for reproducibility.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from deepref.embedding.embedding_generator import EmbeddingGenerator
from deepref.embedding.vector_database import VectorDatabase
from deepref.experiments.run_combine_embeddings_experiments import (
    build_combined_encoder,
    build_encoder1,
    build_encoder2,
    combine_collate_fn,
    load_split_datasets,
)
from deepref.utils.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="generate_embeddings",
)
def main(cfg: DictConfig) -> None:
    """Generate and save train + test VectorDatabases for the configured encoders."""
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    device: str = cfg.device
    cwd = hydra.utils.get_original_cwd()

    # Resolve output directory relative to the original working directory so
    # that Hydra's own output-dir change does not affect the saved paths.
    # Sub-directory per encoder1 model name mirrors the checkpoint layout:
    #   <output_dir>/<enc1_model_name>/<dataset>_<enc1_type>_<enc2_type>_{train|test}.*
    out = Path(cfg.output_dir)
    base_dir = out if out.is_absolute() else Path(cwd) / out
    enc1_model_name = cfg.encoder1.get("model_name", cfg.encoder1.type)
    output_dir = base_dir / enc1_model_name.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    # File-name stem shared by both splits: <dataset>_<enc1>_<enc2>
    stem = f"{cfg.dataset.name}_{cfg.encoder1.type}_{cfg.encoder2.type}"
    train_stem = str(output_dir / f"{stem}_train")
    test_stem  = str(output_dir / f"{stem}_test")

    train_exists = Path(train_stem + VectorDatabase._INDEX_SUFFIX).exists()
    test_exists  = Path(test_stem  + VectorDatabase._INDEX_SUFFIX).exists()

    if train_exists and test_exists and not cfg.force:
        logger.info(
            "Embeddings already exist for '%s' in '%s' — skipping generation.\n"
            "  train: %s.*\n"
            "  test:  %s.*\n"
            "Set force=true to overwrite.",
            stem, output_dir, train_stem, test_stem,
        )
        return

    try:
        # ── Datasets ────────────────────────────────────────────────────────
        train_ds, test_ds = load_split_datasets(cfg.dataset, cwd)
        logger.info(
            "Loaded dataset '%s': train=%d  test=%d  classes=%d",
            cfg.dataset.name, len(train_ds), len(test_ds), len(train_ds.rel2id),
        )

        # ── Encoders ────────────────────────────────────────────────────────
        logger.info("Building encoder1 (%s) …", cfg.encoder1.type)
        encoder1 = build_encoder1(cfg, device)

        logger.info("Building encoder2 (%s) …", cfg.encoder2.type)
        encoder2 = build_encoder2(cfg, device)

        combine = build_combined_encoder(encoder1, encoder2)

        # ── Embedding generation ─────────────────────────────────────────────
        logger.info(
            "Generating train embeddings (batch_size=%d, dim=%d) …",
            cfg.batch_size, combine.model.config.hidden_size,
        )
        train_vdb = EmbeddingGenerator(
            combine,
            train_ds,
            batch_size=cfg.batch_size,
            device=device,
            collate_fn=combine_collate_fn,
        ).generate()
        logger.info("Train VDB ready: %d samples, dim=%d", len(train_vdb), train_vdb.dim)

        logger.info("Generating test embeddings …")
        test_vdb = EmbeddingGenerator(
            combine,
            test_ds,
            batch_size=cfg.batch_size,
            device=device,
            collate_fn=combine_collate_fn,
        ).generate()
        logger.info("Test VDB ready: %d samples, dim=%d", len(test_vdb), test_vdb.dim)

        # ── Optional preprocessing pipeline ─────────────────────────────────
        if cfg.fit_pipeline:
            logger.info(
                "Fitting StandardScaler → PCA(0.95) → L2 Norm on train embeddings …"
            )
            train_vdb.fit_pipeline()
            logger.info(
                "Train VDB after pipeline: dim=%d (was %d)",
                train_vdb.dim, train_vdb._raw_dim,
            )

            logger.info("Applying fitted pipeline to test embeddings …")
            test_vdb.apply_pipeline(train_vdb.pipeline)
            logger.info("Test VDB after pipeline: dim=%d", test_vdb.dim)

        # ── Save ────────────────────────────────────────────────────────────
        train_vdb.save(train_stem)
        logger.info("Saved train VDB → %s.*", train_stem)

        test_vdb.save(test_stem)
        logger.info("Saved test VDB  → %s.*", test_stem)

        logger.info(
            "\nDone. Reload with:\n"
            "  from deepref.embedding.vector_database import VectorDatabase\n"
            "  train_vdb = VectorDatabase.load(%r)\n"
            "  test_vdb  = VectorDatabase.load(%r)",
            train_stem, test_stem,
        )

    finally:
        registry = ModelRegistry()
        for name in list(registry.list_loaded()):
            registry.unload(name)


if __name__ == "__main__":
    main()
