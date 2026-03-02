# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation
```bash
uv sync
# or:
pip install -r requirements.txt && python setup.py develop
```
Requires Python ≥ 3.13.

### Running Tests
```bash
# All tests
pytest deepref/tests/

# Single file or test
pytest deepref/tests/framework/test_sentence_re_trainer.py
pytest deepref/tests/framework/test_sentence_re_trainer.py::TestEarlyStopping::test_disabled_when_patience_zero

# Skip integration tests (require large model downloads / GPU)
pytest deepref/tests/ -m "not integration"
```

### Training
```bash
# Reads/creates deepref/hyperparameters/hyperparams_<dataset>.json automatically
python deepref/framework/train.py --dataset semeval2010
```

### Combine-Embeddings Experiment (Hydra + MLflow)
```bash
# Single run (defaults: relation + bow_sdp, semeval2010)
python deepref/experiments/run_combine_embeddings_experiments.py

# All 4 encoder combinations via multirun
python deepref/experiments/run_combine_embeddings_experiments.py \
    --multirun encoder1=relation,llm encoder2=bow_sdp,verbalized_sdp

# Override individual training params
python deepref/experiments/run_combine_embeddings_experiments.py \
    training.patience=1 training.max_epoch=5
```
MLflow tracking URI and experiment name are set in `deepref/experiments/conf/combine_experiment.yaml`. Run `mlflow ui` to inspect results.

### Hyperparameter Optimization
```bash
python deepref/optimization/bo_optimizer.py -d semeval2010 -m micro_f1 -t 50
# -d: semeval2010 | semeval20181-1 | semeval20181-2 | ddi
# -m: micro_f1 | macro_f1 | acc
# -t: number of Optuna trials
```

### Dataset Download & Preprocessing
```bash
bash benchmark/download_<dataset_name>.sh
# NLP tool selection (spacy is faster; stanza is more accurate):
export NLP_TOOL=spacy   # or stanza
export NLP_MODEL=en_core_web_sm
```
HuggingFace model weights and tokenizer caches are stored in `tmp/`.

---

## Architecture

DeepREF is a sentence-level relation classification framework. The core pipeline converts raw datasets → CSV → preprocessed TXT → training.

### Data Flow
1. **Download** — `benchmark/download_<dataset>.sh` fetches raw files.
2. **Convert** — `deepref/dataset/converters/<dataset>_converter.py` transforms raw data into tab-separated CSV (`benchmark/<dataset>/original/`) with columns: `original_sentence, e1, e2, relation_type, pos_tags, dependencies_labels, ner, sk_entities`. Entity fields are JSON-serialized dicts with `name` and `position`.
3. **Preprocess** — `deepref/dataset/preprocessors/` applies text transforms (see codes below) and writes per-line Python-dict `.txt` files to `benchmark/<dataset>/<preprocessing_str>/`.
4. **Train** — `deepref/framework/train.py` loads TXT files, instantiates encoder + model, trains, and evaluates.

Preprocessing codes: `eb` (entity blinding), `nb` (NER blinding), `d` (digit blinding), `b` (bracket removal), `p` (punctuation removal), `sw` (stopword removal). Combinations are stored as sorted strings, e.g. `"b+d+sw"`.

### Encoder Hierarchy

All encoders ultimately extend `nn.Module`.

```
SentenceEncoder (abstract, deepref/encoder/sentence_encoder.py)
└── LLMEncoder   (deepref/encoder/llm_encoder.py)
    │   Loads AutoModel + AutoTokenizer; registers <e1></e1><e2></e2> tokens;
    │   forward(texts) → L2-normalised average-pool embeddings (B, H).
    │   Model weights cached in tmp/.
    ├── RelationEncoder  (deepref/encoder/relation_encoder.py)
    │     Formats input as: "<e1>head</e1> … <e2>tail</e2> … [MASK]"
    │     forward() → concat of hidden states at <e1>, <e2>, [MASK] → (B, 3H)
    └── VerbalizedSDPEncoder  (deepref/encoder/sdp_encoder.py)
          Inherits both SDPEncoder (spaCy BFS) and LLMEncoder.
          Verbalizes the shortest dependency path, then encodes via LLMEncoder.

SDPEncoder (abstract, deepref/encoder/sdp_encoder.py)
    Provides spaCy en_core_web_trf parsing, BFS shortest-dependency-path,
    and verbalization: "Sentence … | Entity-1: […] | Entity-2: […] | Dependency path: …"
└── BoWSDPEncoder(SDPEncoder, SentenceEncoder)
      No transformer. forward(item) → multi-hot tensor over dep_vocab (len_vocab,).
```

`bert_encoder.py` and `base_bert_encoder.py` are legacy encoders kept for backward compatibility.

### Key Modules

**`deepref/config.py`** — Canonical lists of valid `DATASETS`, `MODELS`, `METRICS`, `PREPROCESSING_TYPES`, `PRETRAIN_WEIGHTS`, and default `HPARAMS`. Any new dataset or model must be registered here.

**`deepref/framework/`**
- `train.py` — `Training` class: full pipeline from preprocessing through evaluation. Entry point for standard training.
- `sentence_re_trainer.py` — `SentenceRETrainer(nn.Module)`: epoch loop, optimizer, warmup scheduler, checkpoint saving. Reads `patience` from `training_parameters` for early stopping.
- `early_stopping.py` — `EarlyStopping(patience)`: `step(improved) -> bool`. `patience=0` disables it.
- `fine_tuner.py` — LoRA-based fine-tuning via HuggingFace `Trainer` + PEFT (alternative to `SentenceRETrainer`).
- `data_loader.py` — `SentenceREDataset` / `SentenceRELoader` for TXT-format data.

**`deepref/model/`**
- `softmax_nn.py` — `SoftmaxNN`: encoder + linear head (standard model).
- `softmax_mlp.py` — `SoftmaxMLP`: encoder + MLP head; used by combine-embeddings experiments.

**`deepref/dataset/`**
- `re_dataset.py` — `REDataset`: PyTorch Dataset over CSV; used by `FineTuner` and combine-embeddings experiments.
- `converters/` — One `DatasetConverter` subclass per dataset; must implement `get_entity_dict()` and `get_sentences()` (generator with `yield`).
- `preprocessors/` — One class per preprocessing type.

**`deepref/experiments/`**
- `run_combine_embeddings_experiments.py` — Hydra-driven experiment that concatenates two encoder embeddings via `CombineEmbeddings`, trains `SoftmaxMLP` using `CombineRETrainer` (a `SentenceRETrainer` subclass), and logs all metrics + params to MLflow.
- `conf/combine_experiment.yaml` — Default config (`training.patience: 3`, `training.max_epoch: 5`, etc.).
- `conf/encoder1/` and `conf/encoder2/` — Hydra config groups for encoder selection.
- `conf/dataset/` — Per-dataset CSV paths.

**`deepref/optimization/`**
- `bo_optimizer.py` — Optuna TPE + Hyperband Bayesian optimization over preprocessing × model × hyperparameter space.

### Hyperparameter Files
`deepref/hyperparameters/hyperparams_<dataset>.json` — auto-created with defaults on first run. Specifies model, pretrain weights, batch size, lr, preprocessing list, and embedding flags.

### Data Formats
- **CSV**: tab-separated; `benchmark/<dataset>/original/`; columns `original_sentence, e1, e2, relation_type, pos_tags, dependencies_labels, ner, sk_entities`.
- **TXT**: one Python dict string per line; keys `token, h, t, relation, pos_tags, deps, ner, sk`; `benchmark/<dataset>/<preprocessing_str>/`.
- **rel2id JSON**: `benchmark/<dataset>/<dataset>_rel2id.json` — relation name → integer id.
- **Checkpoints**: `ckpt/<dataset>_<model>.pth.tar`.
- **Results**: `results/<dataset>/ResultsDeepREF_<dataset>_<datetime>.txt`.

### Adding New Components

**New encoder**: Subclass `LLMEncoder` (or `SentenceEncoder` for non-transformer encoders). Register in `deepref/encoder/__init__.py` and add to `MODELS` in `deepref/config.py`. Add instantiation logic in `deepref/framework/train.py`.

**New dataset**: Add `benchmark/download_<dataset>.sh`, create `deepref/dataset/converters/<dataset>_converter.py` inheriting `DatasetConverter`, add to `DATASETS` in `deepref/config.py`.
