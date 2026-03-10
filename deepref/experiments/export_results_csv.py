"""Export combine-embeddings experiment results from MLflow to a CSV file.

Reads all finished runs from the configured MLflow experiment and writes a
tidy CSV with configuration columns and test metrics, sorted by
``test_micro_f1`` (descending).

Usage
-----
    # Default: reads from local mlruns/, writes results/combine_results.csv
    python deepref/experiments/export_results_csv.py

    # Custom MLflow URI and output path
    python deepref/experiments/export_results_csv.py \\
        --tracking-uri http://localhost:5000 \\
        --experiment combine_embeddings_experiment \\
        --output results/my_results.csv

    # Include failed / in-progress runs too
    python deepref/experiments/export_results_csv.py --all-runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Column ordering
# ---------------------------------------------------------------------------

# Primary metric columns — displayed first so they are immediately visible
METRIC_COLS = [
    "test_micro_f1",
    "test_macro_f1",
    "best_micro_f1",
    "test_micro_p",
    "test_micro_r",
    "test_macro_p",
    "test_macro_r",
]

# Configuration columns — encoder identity, then model, then training hypers
CONFIG_COLS = [
    # Encoder 1
    "encoder1_type",
    "encoder1_model",
    "encoder1_trainable",
    "encoder1_has_cls",
    "encoder1_lr",
    "encoder1_weight_decay",
    "encoder1_warmup_step",
    # Encoder 2
    "encoder2_type",
    "encoder2_model",
    "encoder2_nlp_tool",
    "encoder2_trainable",
    "encoder2_has_cls",
    "encoder2_lr",
    "encoder2_weight_decay",
    "encoder2_warmup_step",
    # Classifier / training
    "dataset",
    "model_type",
    "batch_size",
    "lr",
    "max_epoch",
    "num_mlp_layers",
    "dropout",
    "opt",
    "weight_decay",
    "warmup_step",
    "patience",
    "seed",
    # Vector DB
    "use_vector_db",
    "fit_pipeline",
    "faiss_device",
]

# Run metadata columns appended at the end
META_COLS = [
    "run_name",
    "status",
    "start_time",
    "run_id",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export combine-embeddings MLflow results to CSV")
    p.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: local mlruns/ directory)",
    )
    p.add_argument(
        "--experiment",
        default="combine_embeddings_experiment",
        help="MLflow experiment name (default: combine_embeddings_experiment)",
    )
    p.add_argument(
        "--output",
        default="results/combine_results.csv",
        help="Output CSV path (default: results/combine_results.csv)",
    )
    p.add_argument(
        "--all-runs",
        action="store_true",
        help="Include failed and in-progress runs (default: finished only)",
    )
    return p.parse_args()


def main() -> None:
    try:
        import mlflow
        import pandas as pd
    except ImportError as e:
        print(f"Missing dependency: {e}. Install mlflow and pandas.", file=sys.stderr)
        sys.exit(1)

    args = parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    client = mlflow.tracking.MlflowClient()

    # Resolve experiment
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        print(
            f"Experiment '{args.experiment}' not found. "
            "Run at least one experiment first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fetch runs
    run_filter = None if args.all_runs else "attributes.status = 'FINISHED'"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=run_filter or "",
        max_results=5000,
        order_by=["metrics.test_micro_f1 DESC"],
    )

    if not runs:
        print("No runs found.", file=sys.stderr)
        sys.exit(0)

    # Build a flat dict per run
    records: list[dict] = []
    for run in runs:
        row: dict = {}

        # Metrics
        for col in METRIC_COLS:
            row[col] = run.data.metrics.get(col)

        # Params (all string-typed in MLflow)
        params = run.data.params
        for col in CONFIG_COLS:
            val = params.get(col)
            # Cast numeric-looking strings to float/int for cleaner CSV
            if val is not None:
                try:
                    as_float = float(val)
                    row[col] = int(as_float) if as_float == int(as_float) else as_float
                except ValueError:
                    row[col] = val
            else:
                row[col] = None

        # Metadata
        row["run_name"]   = run.info.run_name
        row["status"]     = run.info.status
        row["start_time"] = pd.Timestamp(run.info.start_time, unit="ms", tz="UTC").strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        row["run_id"] = run.info.run_id

        records.append(row)

    df = pd.DataFrame(records)

    # Build final column order: metrics first, then configs, then meta
    present = set(df.columns)
    ordered_cols = (
        [c for c in METRIC_COLS if c in present]
        + [c for c in CONFIG_COLS if c in present]
        + [c for c in META_COLS if c in present]
        # any extra columns not anticipated above
        + [c for c in df.columns if c not in METRIC_COLS + CONFIG_COLS + META_COLS]
    )
    df = df[ordered_cols]

    # Sort by test_micro_f1 descending (NaN last)
    if "test_micro_f1" in df.columns:
        df = df.sort_values("test_micro_f1", ascending=False, na_position="last")

    # Write CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")

    print(f"Exported {len(df)} run(s) → {out_path}")
    print(f"\nTop-5 by test_micro_f1:")
    display_cols = ["run_name", "test_micro_f1", "test_macro_f1", "encoder1_type", "encoder2_type", "dataset"]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
