"""Fine-tuner experiment runner.

Fine-tunes a single HuggingFace model with LoRA (PEFT) for relation
classification using :class:`~deepref.framework.re_fine_tuner.REFineTuner`,
with MLflow tracking via the HuggingFace Trainer MLflow callback.

Usage
-----
Single run (defaults: bert-base-uncased, semeval2010):
    python deepref/experiments/run_fine_tuner_experiments.py

Different model and dataset:
    python deepref/experiments/run_fine_tuner_experiments.py \\
        model=roberta dataset=ddi

Multirun across models and datasets:
    python deepref/experiments/run_fine_tuner_experiments.py \\
        --multirun model=bert dataset=semeval2010,semeval20181-1,ddi

Override individual training params:
    python deepref/experiments/run_fine_tuner_experiments.py \\
        training.num_train_epochs=5 training.learning_rate=1e-5
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoTokenizer

from deepref.dataset.re_dataset import REDataset
from deepref.framework.re_fine_tuner import REFineTuner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_datasets(
    cfg_dataset: DictConfig,
    cwd: str,
    tokenizer,
) -> tuple[REDataset, REDataset]:
    """Load train and test :class:`REDataset` instances with a unified ``rel2id``.

    Resolves relative paths against the original working directory and handles
    optional ``extra_train_csv_paths`` (e.g. SemEval 2018 sub-tasks).

    Args:
        cfg_dataset: Hydra dataset config node (``name``, ``train_csv_path``,
            ``test_csv_path``, optional ``extra_train_csv_paths``).
        cwd: original working directory (from :func:`hydra.utils.get_original_cwd`).
        tokenizer: HuggingFace tokenizer (already extended with entity tokens).

    Returns:
        ``(train_ds, test_ds)`` pair with a shared ``rel2id`` mapping.
    """
    def resolve(p: str) -> str:
        path = Path(p)
        return str(path if path.is_absolute() else Path(cwd) / path)

    train_paths = [resolve(cfg_dataset.train_csv_path)]
    for extra in cfg_dataset.get("extra_train_csv_paths", []):
        train_paths.append(resolve(extra))

    train_ds = REDataset(
        train_paths if len(train_paths) > 1 else train_paths[0],
        tokenizer=tokenizer,
    )
    test_ds = REDataset(
        resolve(cfg_dataset.test_csv_path),
        tokenizer=tokenizer,
    )

    # Unify relation labels across both splits so class indices are consistent.
    all_relations = sorted(set(train_ds.rel2id) | set(test_ds.rel2id))
    unified_rel2id = {r: i for i, r in enumerate(all_relations)}
    train_ds.rel2id = unified_rel2id
    test_ds.rel2id = unified_rel2id

    logger.info(
        "Loaded '%s': train=%d (from %d file(s))  test=%d  classes=%d",
        cfg_dataset.name,
        len(train_ds),
        len(train_paths),
        len(test_ds),
        len(unified_rel2id),
    )
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="fine_tuner_experiment",
)
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    cwd = hydra.utils.get_original_cwd()

    # ── MLflow ──────────────────────────────────────────────────────────────
    if cfg.mlflow.tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    safe_model = cfg.model.name.replace("/", "_")
    run_name = f"{cfg.dataset.name}__{safe_model}"

    with mlflow.start_run(run_name=run_name):
        use_lora: bool = cfg.use_lora
        params: dict[str, Any] = {
            "dataset": cfg.dataset.name,
            "model_name": cfg.model.name,
            "use_lora": use_lora,
            "num_train_epochs": cfg.training.num_train_epochs,
            "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
            "warmup_steps": cfg.training.warmup_steps,
            "seed": cfg.training.seed,
        }
        if use_lora:
            params.update({
                "lora_r": cfg.lora.r,
                "lora_alpha": cfg.lora.lora_alpha,
                "lora_dropout": cfg.lora.lora_dropout,
                "lora_bias": cfg.lora.bias,
                "lora_task_type": cfg.lora.task_type,
            })
        mlflow.log_params(params)

        try:
            # ── Tokenizer ────────────────────────────────────────────────────
            logger.info("Loading tokenizer: %s", cfg.model.name)
            tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
            n_new = tokenizer.add_special_tokens({
                "additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]
            })

            # ── Datasets ─────────────────────────────────────────────────────
            train_ds, test_ds = load_datasets(cfg.dataset, cwd, tokenizer)

            # ── Training arguments + LoRA config ─────────────────────────────
            output_dir = os.path.join(
                cwd, "ckpt", "fine_tuner", f"{cfg.dataset.name}_{safe_model}"
            )
            training_parameters: dict[str, Any] = {
                **OmegaConf.to_container(cfg.training, resolve=True),
                "output_dir": output_dir,
            }
            lora_config_parameters: dict[str, Any] | None = (
                OmegaConf.to_container(cfg.lora, resolve=True)
                if use_lora
                else None
            )

            # ── Fine-tuner ───────────────────────────────────────────────────
            mode = "LoRA" if use_lora else "full fine-tuning"
            logger.info("Instantiating REFineTuner (%s) …", mode)
            fine_tuner = REFineTuner(
                model_name=cfg.model.name,
                train_dataset=train_ds,
                test_dataset=test_ds,
                n_new=n_new,
                training_parameters=training_parameters,
                lora_config_parameters=lora_config_parameters,
            )

            # ── Training ─────────────────────────────────────────────────────
            # The HuggingFace Trainer's MLflow callback detects the active run
            # started above and logs per-epoch training metrics automatically.
            logger.info("Starting fine-tuning …")
            fine_tuner.finetune_model()

            # ── Test evaluation ──────────────────────────────────────────────
            logger.info("=== Final evaluation on test set ===")
            eval_results = fine_tuner.trainer.evaluate()
            logger.info("Eval results: %s", eval_results)

            mlflow.log_metrics({
                "test_precision": eval_results.get("eval_precision", 0.0),
                "test_recall":    eval_results.get("eval_recall",    0.0),
                "test_f1":        eval_results.get("eval_f1",        0.0),
                "test_loss":      eval_results.get("eval_loss",      0.0),
            })

            # ── Persist LoRA adapter ─────────────────────────────────────────
            fine_tuner.save_model()
            mlflow.log_artifact(output_dir, artifact_path="model")

            mlflow.set_tag("status", "success")
            logger.info(
                "Run '%s' finished — test_f1: %.4f",
                run_name,
                eval_results.get("eval_f1", 0.0),
            )

        except Exception:
            import traceback
            logger.error("Run '%s' failed:\n%s", run_name, traceback.format_exc())
            mlflow.set_tag("status", "failed")
            raise


if __name__ == "__main__":
    main()
