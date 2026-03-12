"""
Combine-Embeddings experiment runner.

Trains a classifier on top of one or two encoders using SentenceRETrainer as
the training engine.  When two encoders are used their embeddings are
concatenated via :class:`CombineEmbeddings`; when only one encoder is used a
:class:`SingleEncoderWrapper` feeds it directly into the classifier.

Supported classifiers (``training.model_type``):
  softmax_mlp          — MLP with softmax head (gradient-based, default)
  combine_re_classifier — dual-branch MLP (BERT branch 2034→1152, Qwen branch 4096→128, merged 1280→N)
  xgboost              — XGBoost multi-class classifier (sklearn API)
  lightgbm             — LightGBM multi-class classifier (sklearn API)

Dual-encoder combinations (driven by Hydra multirun):
  encoder1=relation         encoder2=bow_sdp
  encoder1=relation         encoder2=verbalized_sdp
  encoder1=llm              encoder2=bow_sdp
  encoder1=llm              encoder2=verbalized_sdp
  encoder1=bert_entity_cls  encoder2=bow_sdp
  encoder1=bert_entity_cls  encoder2=verbalized_sdp

Single-encoder mode (set encoder2=none):
  encoder1=relation         encoder2=none
  encoder1=llm              encoder2=none
  encoder1=bert_entity      encoder2=none   (CLS disabled: [E1]+[E2] → 2H)
  encoder1=bert_entity_cls  encoder2=none   (CLS enabled:  [CLS]+[E1]+[E2] → 3H)

Usage
-----
Single run (defaults: relation + bow_sdp, semeval2010):
    python deepref/experiments/run_combine_embeddings_experiments.py

All 4 dual-encoder combinations via multirun:
    python deepref/experiments/run_combine_embeddings_experiments.py \\
        --multirun encoder1=relation,llm encoder2=bow_sdp,verbalized_sdp

Single-encoder run (encoder1 only):
    python deepref/experiments/run_combine_embeddings_experiments.py \\
        encoder2=none

All single + dual combinations via multirun:
    python deepref/experiments/run_combine_embeddings_experiments.py \\
        --multirun encoder1=relation,llm encoder2=bow_sdp,verbalized_sdp,none

Cross dataset × encoder multirun:
    python deepref/experiments/run_combine_embeddings_experiments.py \\
        --multirun \\
        dataset=semeval2010,ddi \\
        encoder1=relation,llm \\
        encoder2=bow_sdp,verbalized_sdp,none

XGBoost / LightGBM run:
    python deepref/experiments/run_combine_embeddings_experiments.py \\
        training.model_type=xgboost
VectorDatabase mode (pre-compute embeddings once, train MLP head on tensors):
    python deepref/experiments/run_combine_embeddings_experiments.py \\
        vector_db.enabled=true

    Optionally save the generated VectorDatabases to disk:
    python deepref/experiments/run_combine_embeddings_experiments.py \\
        vector_db.enabled=true vector_db.save_dir=embeddings/

    With PCA preprocessing (StandardScaler → PCA(0.95) → L2 Norm):
    python deepref/experiments/run_combine_embeddings_experiments.py \\
        vector_db.enabled=true vector_db.fit_pipeline=true

    PCA-transformed VDBs are saved alongside raw ones with a ``_pca`` infix
    in the filename stem when ``vector_db.save_dir`` is also set.
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
import mlflow
import numpy as np
import torch
import random
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from deepref.dataset.combine_re_dataset import CombineREDataset
from deepref.embedding.embedding_generator import EmbeddingGenerator
from deepref.embedding.vector_database import VectorDatabase
from deepref.encoder.bert_entity_encoder import BertEntityEncoder
from deepref.encoder.combine_embeddings import CombineEmbeddings
from deepref.encoder.llm_encoder import LLMEncoder
from deepref.encoder.relation_encoder import RelationEncoder
from deepref.encoder.sdp_encoder import BoWSDPEncoder, VerbalizedSDPEncoder
from deepref.encoder.single_encoder_wrapper import SingleEncoderWrapper
from deepref.framework.combine_re_trainer import CombineRETrainer
from deepref.framework.vector_db_re_trainer import VectorDBRETrainer
from deepref.model.combine_re_classifier import CombineREClassifier
from deepref.model.sklearn_re_classifier import SklearnREClassifier
from deepref.model.softmax_mlp import SoftmaxMLP
from deepref.nlp.nlp_tool import NLPTool
from deepref.utils.focal_loss import FocalLossLabelSmoothing
from deepref.utils.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def combine_collate_fn(
    batch: list[tuple[dict, torch.Tensor]],
) -> dict[str, Any]:
    """Collate ``(item_dict, label)`` pairs into a batch dict.

    Returns a dict with:
    * ``"labels"``: ``(B,)`` long tensor — compatible with
      :meth:`SentenceRETrainer.iterate_loader`.
    * ``"items"``: plain Python list of item dicts — passed as the
      ``items`` keyword argument to :class:`CombineEmbeddings`.
    """
    items, labels = zip(*batch)
    return {"labels": torch.stack(labels), "items": list(items)}


def load_split_datasets(
    cfg_dataset: DictConfig,
    cwd: str,
) -> tuple[CombineREDataset, CombineREDataset]:
    """Load pre-split train and test datasets from the benchmark CSV files.

    Builds a unified ``rel2id`` from the union of both splits so that class
    indices are consistent across train and test.
    """
    def resolve(p: str) -> str:
        path = Path(p)
        return str(path if path.is_absolute() else Path(cwd) / path)

    train_paths = [resolve(cfg_dataset.train_csv_path)]
    for extra in cfg_dataset.get("extra_train_csv_paths", []):
        train_paths.append(resolve(extra))

    train_ds = CombineREDataset(train_paths if len(train_paths) > 1 else train_paths[0])
    test_ds = CombineREDataset(resolve(cfg_dataset.test_csv_path))

    # Unify relation labels across both splits
    all_relations = sorted(set(train_ds.rel2id) | set(test_ds.rel2id))
    unified_rel2id = {r: i for i, r in enumerate(all_relations)}
    train_ds.rel2id = unified_rel2id
    test_ds.rel2id = unified_rel2id

    logger.info(
        "Loaded '%s': train=%d (from %d file(s))  test=%d  classes=%d",
        cfg_dataset.name, len(train_ds), len(train_paths), len(test_ds), len(unified_rel2id),
    )
    return train_ds, test_ds

# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------

def build_encoder1(cfg: DictConfig, device: str) -> nn.Module:
    """Instantiate the first encoder from Hydra config."""
    enc = cfg.encoder1
    if enc.type == "relation":
        return RelationEncoder(
            model_name=enc.model_name,
            max_length=enc.max_length,
            device=device,
            trainable=enc.get("trainable", False),
            attn_implementation=enc.get("attn_implementation", "eager"),
        )
    if enc.type == "bert_entity":
        return BertEntityEncoder(
            model_name=enc.model_name,
            max_length=enc.max_length,
            has_cls_embedding=enc.get("has_cls_embedding", False),
            device=device,
            trainable=enc.get("trainable", False),
            attn_implementation=enc.get("attn_implementation", "eager"),
        )
    if enc.type == "llm":
        return LLMEncoder(
            model_name=enc.model_name,
            max_length=enc.max_length,
            device=device,
            trainable=enc.trainable,
            attn_implementation=enc.attn_implementation,
        )
    if enc.type == "verbalized_sdp":
        nlp_tool_name = enc.get("nlp_tool", "spacy")
        nlp_tool = build_nlp_tool(nlp_tool_name)
        return VerbalizedSDPEncoder(nlp_tool=nlp_tool, model_name=enc.model_name, device=device)
    raise ValueError(f"Unknown encoder1 type: {enc.type!r}")


def build_nlp_tool(name: str) -> NLPTool:
    """Instantiate an NLP tool by name (``'spacy'`` or ``'stanza'``)."""
    if name == "spacy":
        from deepref.nlp.spacy_nlp_tool import SpacyNLPTool
        return SpacyNLPTool()
    if name == "stanza":
        from deepref.nlp.stanza_nlp_tool import StanzaNLPTool
        return StanzaNLPTool()
    raise ValueError(f"Unknown NLP tool: {name!r}. Choose 'spacy' or 'stanza'.")


def build_encoder2(cfg: DictConfig, device: str) -> nn.Module | None:
    """Instantiate the second encoder from Hydra config, or return ``None``.

    Returns ``None`` when ``cfg.encoder2.type == "none"``, enabling
    single-encoder mode via :class:`SingleEncoderWrapper` in
    :func:`build_combined_encoder`.
    """
    enc = cfg.encoder2
    if enc.type == "none":
        return None
    if enc.type == "bert_entity":
        return BertEntityEncoder(
            model_name=enc.model_name,
            max_length=enc.max_length,
            has_cls_embedding=enc.get("has_cls_embedding", False),
            device=device,
            trainable=enc.get("trainable", False),
            attn_implementation=enc.get("attn_implementation", "eager"),
        )
    nlp_tool_name = enc.get("nlp_tool", "spacy")
    nlp_tool = build_nlp_tool(nlp_tool_name)
    if enc.type == "bow_sdp":
        return BoWSDPEncoder(nlp_tool=nlp_tool)
    if enc.type == "verbalized_sdp":
        return VerbalizedSDPEncoder(nlp_tool=nlp_tool, model_name=enc.model_name, device=device)
    raise ValueError(f"Unknown encoder2 type: {enc.type!r}")


def build_combined_encoder(
    encoder1: nn.Module,
    encoder2: nn.Module | None,
) -> nn.Module:
    """Return a :class:`CombineEmbeddings` or :class:`SingleEncoderWrapper`.

    Args:
        encoder1: always required.
        encoder2: second encoder, or ``None`` for single-encoder mode.

    Returns:
        :class:`CombineEmbeddings` when both encoders are provided;
        :class:`SingleEncoderWrapper` wrapping *encoder1* when *encoder2*
        is ``None``.
    """
    if encoder2 is None:
        logger.info("Single-encoder mode — wrapping encoder1 with SingleEncoderWrapper")
        return SingleEncoderWrapper(encoder1)
    logger.info("Dual-encoder mode — combining encoders with CombineEmbeddings")
    return CombineEmbeddings(encoder1, encoder2)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    cfg: DictConfig,
    sentence_encoder: nn.Module | None,
    num_class: int,
    rel2id: dict,
    hidden_size: int | None = None,
) -> nn.Module:
    """Instantiate the classifier from ``cfg.training.model_type``.

    Args:
        cfg: full Hydra config.
        sentence_encoder: combined encoder to embed sentences.  May be
            ``None`` when training on pre-computed VDB embeddings.
        num_class: number of relation classes.
        rel2id: relation-name → class-index mapping.
        hidden_size: embedding dimension used when ``sentence_encoder`` is
            ``None`` (VDB mode without encoder loading).

    Returns:
        A :class:`~deepref.model.softmax_mlp.SoftmaxMLP` for
        ``model_type="softmax_mlp"`` or a :class:`SklearnREClassifier`
        for ``"xgboost"`` / ``"lightgbm"``.
    """
    model_type: str = cfg.training.get("model_type", "softmax_mlp")

    if model_type == "softmax_mlp":
        return SoftmaxMLP(
            sentence_encoder=sentence_encoder,
            num_class=num_class,
            rel2id=rel2id,
            dropout=cfg.training.dropout,
            num_layers=cfg.training.num_mlp_layers,
            hidden_size=hidden_size,
        )

    if model_type == "combine_re_classifier":
        return CombineREClassifier(
            sentence_encoder=sentence_encoder,
            num_class=num_class,
            rel2id=rel2id,
        )

    if model_type in ("xgboost", "lightgbm"):
        return SklearnREClassifier(
            sentence_encoder=sentence_encoder,
            num_class=num_class,
            rel2id=rel2id,
            model_type=model_type,
        )

    raise ValueError(
        f"Unknown model_type: {model_type!r}. "
        "Choose 'softmax_mlp', 'combine_re_classifier', 'xgboost', or 'lightgbm'."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    rel2id: dict[str, int],
    run_name: str,
) -> None:
    """Plot a confusion matrix and log it as an MLflow PNG artifact.

    The figure is written to a temporary file, logged under the MLflow
    ``figures/`` artifact path, then deleted.

    Args:
        y_true:   ground-truth class indices.
        y_pred:   predicted class indices.
        rel2id:   relation-name → class-index mapping (used to derive labels).
        run_name: used as the figure title.
    """
    import tempfile

    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe in all environments
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    id2rel = {v: k for k, v in rel2id.items()}
    labels = sorted(id2rel.keys())
    class_names = [id2rel[i] for i in labels]
    n = len(labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Scale figure so each cell is roughly 0.6 in × 0.6 in; minimum 8 × 8.
    fig_size = max(8, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm,
        annot=n <= 30,   # skip per-cell numbers when the matrix is very large
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(f"Confusion matrix — {run_name}", fontsize=13, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=max(6, 10 - n // 5))
    plt.yticks(rotation=0,  fontsize=max(6, 10 - n // 5))
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig_path = f.name
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    mlflow.log_artifact(fig_path, artifact_path="figures")
    os.remove(fig_path)
    logger.info("Confusion matrix logged to MLflow artifacts (figures/)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="combine_experiment",
)
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.training.seed)
    device: str = cfg.device

    # ── MLflow ──────────────────────────────────────────────────────────────
    if cfg.mlflow.tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_name = (
        f"{cfg.dataset.name}"
        f"__{cfg.encoder1.type}"
        f"__{cfg.encoder2.type}"
        f"__{cfg.training.get('model_type', 'softmax_mlp')}"
    )

    set_seed(cfg.training.seed)

    with mlflow.start_run(run_name=run_name):
        # Per-encoder training overrides (fall back to global training params)
        enc1_training = cfg.encoder1.get("training", {})
        enc2_training = cfg.encoder2.get("training", {}) if cfg.encoder2.type != "none" else {}
        enc1_lr     = enc1_training.get("lr",           cfg.training.lr)                    if enc1_training else cfg.training.lr
        enc1_wd     = enc1_training.get("weight_decay", cfg.training.get("weight_decay", 0.0)) if enc1_training else cfg.training.get("weight_decay", 0.0)
        enc1_warmup = enc1_training.get("warmup_step",  0)                                  if enc1_training else 0
        enc2_lr     = enc2_training.get("lr",           cfg.training.lr)                    if enc2_training else cfg.training.lr
        enc2_wd     = enc2_training.get("weight_decay", cfg.training.get("weight_decay", 0.0)) if enc2_training else cfg.training.get("weight_decay", 0.0)
        enc2_warmup = enc2_training.get("warmup_step",  0)                                  if enc2_training else 0

        model_type: str = cfg.training.get("model_type", "softmax_mlp")
        use_vector_db: bool = cfg.vector_db.enabled

        mlflow.log_params({
            "dataset": cfg.dataset.name,
            "model_type": model_type,
            "encoder1_type": cfg.encoder1.type,
            "encoder1_model": cfg.encoder1.get("model_name", "n/a"),
            "encoder1_trainable": cfg.encoder1.get("trainable", False),
            "encoder1_has_cls": cfg.encoder1.get("has_cls_embedding", False),
            "encoder1_lr": enc1_lr,
            "encoder1_weight_decay": enc1_wd,
            "encoder1_warmup_step": enc1_warmup,
            "encoder2_type": cfg.encoder2.type,
            "encoder2_model": cfg.encoder2.get("model_name", "n/a"),
            "encoder2_nlp_tool": cfg.encoder2.get("nlp_tool", "n/a"),
            "encoder2_trainable": cfg.encoder2.get("trainable", False),
            "encoder2_has_cls": cfg.encoder2.get("has_cls_embedding", False) if cfg.encoder2.type != "none" else "n/a",
            "encoder2_lr": enc2_lr if cfg.encoder2.type != "none" else "n/a",
            "encoder2_weight_decay": enc2_wd if cfg.encoder2.type != "none" else "n/a",
            "encoder2_warmup_step": enc2_warmup if cfg.encoder2.type != "none" else "n/a",
            "batch_size": cfg.training.batch_size,
            "lr": cfg.training.lr,
            "max_epoch": cfg.training.max_epoch,
            "num_mlp_layers": cfg.training.get("num_mlp_layers", "n/a"),
            "dropout": cfg.training.get("dropout", "n/a"),
            "opt": cfg.training.opt,
            "seed": cfg.training.seed,
            "patience": cfg.training.get("patience", 0),
            "use_vector_db": use_vector_db,
            "fit_pipeline": cfg.vector_db.get("fit_pipeline", False),
            "faiss_device": cfg.vector_db.get("faiss_device", None) or device,
        })

        try:
            # ── Dataset ─────────────────────────────────────────────────────
            cwd = hydra.utils.get_original_cwd()
            train_dataset, test_dataset = load_split_datasets(cfg.dataset, cwd)
            num_class = len(train_dataset.rel2id)

            # ── Trainer ─────────────────────────────────────────────────────
            enc1_model_name = cfg.encoder1.get("model_name", cfg.encoder1.type)
            ckpt_dir = os.path.join(
                cwd,
                "ckpt",
                enc1_model_name.replace("/", "_"),
            )
            ckpt_path = os.path.join(
                ckpt_dir,
                f"{cfg.dataset.name}_{cfg.encoder1.type}_{cfg.encoder2.type}_{model_type}.pth",
            )

            training_parameters = {
                "max_epoch": cfg.training.max_epoch,
                "criterion": FocalLossLabelSmoothing(),
                "lr": cfg.training.lr,
                "batch_size": cfg.training.batch_size,
                "opt": cfg.training.opt,
                "weight_decay": cfg.training.get("weight_decay", 0.0),
                "warmup_step": cfg.training.get("warmup_step", 0),
                "patience": cfg.training.get("patience", 0),
                "encoder1_lr": enc1_lr,
                "encoder1_weight_decay": enc1_wd,
                "encoder1_warmup_step": enc1_warmup,
                "encoder2_lr": enc2_lr,
                "encoder2_weight_decay": enc2_wd,
                "encoder2_warmup_step": enc2_warmup,
            }

            if use_vector_db:
                # ── VectorDB path: load from disk or generate then save ──────
                fit_pipeline: bool = cfg.vector_db.get("fit_pipeline", False)
                vdb_batch_size = cfg.vector_db.get("batch_size", None) or cfg.training.batch_size
                vdb_save_dir   = cfg.vector_db.get("save_dir", None)

                enc1_tag = cfg.encoder1.type + ("_cls" if cfg.encoder1.get("has_cls_embedding", False) else "")
                enc2_tag = cfg.encoder2.type + ("_cls" if cfg.encoder2.get("has_cls_embedding", False) else "")
                vdb_stem = f"{cfg.dataset.name}_{enc1_tag}_{enc2_tag}"
                enc1_model_name = cfg.encoder1.get("model_name", cfg.encoder1.type)

                save_dir = None
                train_stem = test_stem = None
                pca_train_stem = pca_test_stem = None
                train_exists = test_exists = False
                pca_train_exists = pca_test_exists = False

                if vdb_save_dir:
                    save_dir = (
                        Path(vdb_save_dir)
                        if Path(vdb_save_dir).is_absolute()
                        else Path(cwd) / vdb_save_dir
                    ) / enc1_model_name.replace("/", "_")
                    train_stem = str(save_dir / f"{vdb_stem}_train")
                    test_stem  = str(save_dir / f"{vdb_stem}_test")
                    pca_train_stem = str(save_dir / f"{vdb_stem}_pca_train")
                    pca_test_stem  = str(save_dir / f"{vdb_stem}_pca_test")
                    train_exists = Path(train_stem + VectorDatabase._INDEX_SUFFIX).exists()
                    test_exists  = Path(test_stem  + VectorDatabase._INDEX_SUFFIX).exists()
                    pca_train_exists = fit_pipeline and Path(pca_train_stem + VectorDatabase._INDEX_SUFFIX).exists()
                    pca_test_exists  = fit_pipeline and Path(pca_test_stem  + VectorDatabase._INDEX_SUFFIX).exists()

                faiss_device: str = cfg.vector_db.get("faiss_device", device)

                # Encoders are only needed when VDBs must be generated.
                vdb_cached = (pca_train_exists or train_exists) and (pca_test_exists or test_exists)
                combine = None
                if not vdb_cached:
                    logger.info("Building encoder1 (%s) …", cfg.encoder1.type)
                    encoder1 = build_encoder1(cfg, device)
                    logger.info("Building encoder2 (%s) …", cfg.encoder2.type)
                    encoder2 = build_encoder2(cfg, device)
                    combine = build_combined_encoder(encoder1, encoder2)
                else:
                    logger.info(
                        "VDB found on disk — skipping encoder loading."
                    )

                # ── Load or generate train VDB ───────────────────────────────
                if pca_train_exists:
                    logger.info("Loading PCA train VDB from %s.*", pca_train_stem)
                    train_vdb = VectorDatabase.load(pca_train_stem, device=faiss_device)
                elif train_exists:
                    logger.info("Loading train VDB from %s.*", train_stem)
                    train_vdb = VectorDatabase.load(train_stem, device=faiss_device)
                else:
                    logger.info(
                        "Generating train VectorDatabase (batch_size=%d) …", vdb_batch_size
                    )
                    train_vdb = EmbeddingGenerator(
                        combine, train_dataset, batch_size=vdb_batch_size, device=device,
                        faiss_device=faiss_device, collate_fn=combine_collate_fn,
                    ).generate()
                    if save_dir:
                        save_dir.mkdir(parents=True, exist_ok=True)
                        train_vdb.save(train_stem)
                        logger.info("Saved train VDB → %s.*", train_stem)

                # ── Load or generate test VDB ────────────────────────────────
                if pca_test_exists:
                    logger.info("Loading PCA test VDB from %s.*", pca_test_stem)
                    test_vdb = VectorDatabase.load(pca_test_stem, device=faiss_device)
                elif test_exists:
                    logger.info("Loading test VDB from %s.*", test_stem)
                    test_vdb = VectorDatabase.load(test_stem, device=faiss_device)
                else:
                    logger.info("Generating test VectorDatabase …")
                    test_vdb = EmbeddingGenerator(
                        combine, test_dataset, batch_size=vdb_batch_size, device=device,
                        faiss_device=faiss_device, collate_fn=combine_collate_fn,
                    ).generate()
                    if save_dir:
                        save_dir.mkdir(parents=True, exist_ok=True)
                        test_vdb.save(test_stem)
                        logger.info("Saved test VDB → %s.*", test_stem)

                # ── PCA pipeline (fit on train, apply to test) ───────────────
                if fit_pipeline and train_vdb.pipeline is None:
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
                    if save_dir:
                        save_dir.mkdir(parents=True, exist_ok=True)
                        train_vdb.save(pca_train_stem)
                        test_vdb.save(pca_test_stem)
                        logger.info(
                            "Saved PCA VDBs → %s.*  /  %s.*",
                            pca_train_stem, pca_test_stem,
                        )

                # ── Model ────────────────────────────────────────────────────
                # Use VDB dim directly — no need for the encoder's config.
                vdb_dim = train_vdb.dim
                logger.info("VDB embedding dim: %d", vdb_dim)
                model = build_model(
                    cfg, combine, num_class, train_dataset.rel2id,
                    hidden_size=vdb_dim,
                )
                logger.info("Classifier: %s", model_type)

                trainer = VectorDBRETrainer(
                    model=model,
                    train_vdb=train_vdb,
                    test_vdb=test_vdb,
                    ckpt=ckpt_path,
                    training_parameters=training_parameters,
                )
            else:
                # ── Traditional path: embed on the fly during each training step ──
                logger.info("Building encoder1 (%s) …", cfg.encoder1.type)
                encoder1 = build_encoder1(cfg, device)

                logger.info("Building encoder2 (%s) …", cfg.encoder2.type)
                encoder2 = build_encoder2(cfg, device)

                combine = build_combined_encoder(encoder1, encoder2)

                model = build_model(cfg, combine, num_class, train_dataset.rel2id)
                logger.info("Classifier: %s", model_type)

                trainer = CombineRETrainer(
                    model=model,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    ckpt=ckpt_path,
                    training_parameters=training_parameters,
                )

            # ── Training ────────────────────────────────────────────────────
            best_micro_f1 = trainer.train_model(metric="micro_f1")

            mlflow.log_metric("best_micro_f1", best_micro_f1)

            # ── Test evaluation ─────────────────────────────────────────────
            logger.info("=== Test evaluation ===")
            test_result, test_preds, test_labels = trainer.eval_model(trainer.test_loader)
            mlflow.log_metrics({
                "test_micro_p":  test_result["micro_p"],
                "test_micro_r":  test_result["micro_r"],
                "test_micro_f1": test_result["micro_f1"],
                "test_macro_p":  test_result["macro_p"],
                "test_macro_r":  test_result["macro_r"],
                "test_macro_f1": test_result["macro_f1"],
            })
            logger.info(
                "Test metrics — micro_f1: %.4f  macro_f1: %.4f",
                test_result["micro_f1"], test_result["macro_f1"],
            )

            _log_confusion_matrix(test_labels, test_preds, train_dataset.rel2id, run_name)

            mlflow.set_tag("status", "success")

            logger.info("Run '%s' finished — best micro_f1: %.4f", run_name, best_micro_f1)

        except Exception:
            import traceback
            logger.error("Run '%s' failed:\n%s", run_name, traceback.format_exc())
            mlflow.set_tag("status", "failed")
            raise
        finally:
            registry = ModelRegistry()
            for name in list(registry.list_loaded()):
                registry.unload(name)


if __name__ == "__main__":
    main()
