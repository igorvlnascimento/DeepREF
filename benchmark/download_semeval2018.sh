#!/usr/bin/env bash
set -euo pipefail

mkdir -p benchmark/raw_semeval20181-1/Train benchmark/raw_semeval20181-1/Test
mkdir -p benchmark/raw_semeval20181-2/Train benchmark/raw_semeval20181-2/Test

# Download from HuggingFace (DFKI-SLT/SemEval2018_Task7) and convert to JSON.
# Requires: huggingface_hub, pandas, pyarrow (all available via uv).
uv run python - <<'PYEOF'
import json
import pandas as pd
from huggingface_hub import hf_hub_download

REPO     = "DFKI-SLT/SemEval2018_Task7"
REVISION = "refs/convert/parquet"

SPLITS = {
    "Subtask_1_1": {
        "train": "benchmark/raw_semeval20181-1/Train/train.json",
        "test":  "benchmark/raw_semeval20181-1/Test/test.json",
    },
    "Subtask_1_2": {
        "train": "benchmark/raw_semeval20181-2/Train/train.json",
        "test":  "benchmark/raw_semeval20181-2/Test/test.json",
    },
}

def to_python(obj):
    """Recursively convert numpy/pandas types to plain Python types."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if hasattr(obj, "tolist"):        # numpy scalar or array → Python native
        return to_python(obj.tolist())
    return obj

for subtask, splits in SPLITS.items():
    for split, out_path in splits.items():
        parquet_path = hf_hub_download(
            repo_id=REPO,
            filename=f"{subtask}/{split}/0000.parquet",
            repo_type="dataset",
            revision=REVISION,
        )
        df = pd.read_parquet(parquet_path)
        records = to_python(df.to_dict(orient="records"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"Written {len(records)} records -> {out_path}")
PYEOF

uv run python deepref/dataset/preprocessor/semeval2018_preprocessor.py --path benchmark/raw_semeval20181-1/ 2>&1 | sed -u 's/^/[1.1] /' &
uv run python deepref/dataset/preprocessor/semeval2018_preprocessor.py --path benchmark/raw_semeval20181-2/ 2>&1 | sed -u 's/^/[1.2] /' &

wait
