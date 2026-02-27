"""
Experiment runner: PromptEntityEncoder × all datasets × pretrain weights.

Runs every combination of dataset and pre-trained model supported by
PromptEntityEncoder, tracking parameters and metrics with MLflow.

Usage
-----
    python deepref/experiments/run_prompt_encoder_experiments.py [options]

Options
-------
    --datasets      Space-separated list of datasets to run (default: all).
    --pretrain      Space-separated list of pretrain weights to run (default: all).
    --preprocessing Space-separated list of preprocessing types (default: none).
    --max_epoch     Number of training epochs (default: 3).
    --batch_size    Batch size (default: 16).
    --lr            Learning rate (default: 2e-5).
    --max_length    Maximum tokenized sequence length (default: 128).
    --experiment    MLflow experiment name (default: "prompt_encoder_combinations").
    --tracking_uri  MLflow tracking URI (default: local ./mlruns).
    --dry_run       Print combinations without executing training.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import traceback
from typing import Any

import mlflow

from deepref import config
from deepref.framework.train import Training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Pretrain weights that expose a [MASK] / <mask> token required by PromptEntityEncoder.
PROMPT_ENCODER_PRETRAIN_WEIGHTS: list[str] = [
    "bert-base-uncased",
    "dmis-lab/biobert-v1.1",
    "allenai/scibert_scivocab_uncased",
]


def generate_combinations(
    datasets: list[str],
    pretrain_weights: list[str],
    preprocessing_list: list[list[str]],
) -> list[dict[str, Any]]:
    """Return all (dataset, pretrain, preprocessing) combinations as dicts."""
    combos = []
    for dataset, pretrain, preprocessing in itertools.product(
        datasets, pretrain_weights, preprocessing_list
    ):
        combos.append(
            {
                "dataset": dataset,
                "pretrain": pretrain,
                "preprocessing": preprocessing,
            }
        )
    return combos


def build_hparams(
    pretrain: str,
    preprocessing: list[str],
    max_epoch: int,
    batch_size: int,
    lr: float,
    max_length: int,
) -> dict[str, Any]:
    """Build a hyperparameter dict compatible with Training()."""
    return {
        "model": "prompt_encoder",
        "pretrain": pretrain,
        "batch_size": batch_size,
        "preprocessing": preprocessing,
        "lr": lr,
        "position_embed": 0,
        "pos_tags_embed": 0,
        "deps_embed": 0,
        "sk_embed": 0,
        "sdp_embed": 0,
        "max_length": max_length,
        "max_epoch": max_epoch,
    }


def run_experiment(
    combo: dict[str, Any],
    max_epoch: int,
    batch_size: int,
    lr: float,
    max_length: int,
    mlflow_experiment: str,
) -> dict[str, Any] | None:
    """Run a single (dataset, pretrain, preprocessing) experiment and log to MLflow."""
    dataset = combo["dataset"]
    pretrain = combo["pretrain"]
    preprocessing = combo["preprocessing"]
    preprocessing_str = "_".join(sorted(preprocessing)) if preprocessing else "original"

    run_name = f"{dataset}__{pretrain.replace('/', '__')}__{preprocessing_str}"
    logger.info("Starting run: %s", run_name)

    hparams = build_hparams(pretrain, preprocessing, max_epoch, batch_size, lr, max_length)

    with mlflow.start_run(run_name=run_name):
        # Log all hyperparameters
        mlflow.log_params(
            {
                "dataset": dataset,
                "model": "prompt_encoder",
                "pretrain": pretrain,
                "preprocessing": preprocessing_str,
                "batch_size": batch_size,
                "lr": lr,
                "max_length": max_length,
                "max_epoch": max_epoch,
            }
        )

        try:
            training = Training(dataset, hparams)
            result = training.train()
        except Exception:
            logger.error("Run %s failed:\n%s", run_name, traceback.format_exc())
            mlflow.set_tag("status", "failed")
            return None

        # Log evaluation metrics
        metrics_to_log = {k: v for k, v in result.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(metrics_to_log)
        mlflow.set_tag("status", "success")

    logger.info("Finished run: %s — metrics: %s", run_name, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all PromptEntityEncoder × dataset combinations."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=config.DATASETS,
        choices=config.DATASETS,
        help="Datasets to include (default: all).",
    )
    parser.add_argument(
        "--pretrain",
        nargs="+",
        default=PROMPT_ENCODER_PRETRAIN_WEIGHTS,
        help="Pre-trained model paths/names (default: BERT-family weights).",
    )
    parser.add_argument(
        "--preprocessing",
        nargs="*",
        default=[[]],
        help=(
            "Preprocessing combinations to include. Pass multiple times for "
            "multiple combos, e.g. --preprocessing sw p.  Default: no preprocessing."
        ),
    )
    parser.add_argument("--max_epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--experiment",
        default="prompt_encoder_combinations",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--tracking_uri",
        default=None,
        help="MLflow tracking URI (default: local ./mlruns).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print combinations without running training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    # Build preprocessing list: always include the "no preprocessing" baseline
    preprocessing_list: list[list[str]] = [[]]
    if args.preprocessing and args.preprocessing != [[]]:
        preprocessing_list.append(args.preprocessing)

    combos = generate_combinations(args.datasets, args.pretrain, preprocessing_list)
    total = len(combos)
    logger.info(
        "Running %d combinations: %d datasets × %d pretrain weights × %d preprocessing configs",
        total,
        len(args.datasets),
        len(args.pretrain),
        len(preprocessing_list),
    )

    if args.dry_run:
        for i, combo in enumerate(combos, 1):
            print(f"[{i}/{total}] {combo}")
        return

    results: dict[str, Any] = {}
    for i, combo in enumerate(combos, 1):
        logger.info("[%d/%d] %s", i, total, combo)
        result = run_experiment(
            combo,
            max_epoch=args.max_epoch,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            mlflow_experiment=args.experiment,
        )
        key = f"{combo['dataset']}__{combo['pretrain']}__{combo['preprocessing']}"
        results[key] = result

    successes = sum(1 for v in results.values() if v is not None)
    logger.info("Completed %d/%d runs successfully.", successes, total)


if __name__ == "__main__":
    main(sys.argv[1:])
