# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation
```bash
pip install -r requirements.txt
python setup.py develop
# or with uv:
uv sync
```

### Running Tests
```bash
# Run all tests
pytest deepref/tests/

# Run a specific test file
pytest deepref/tests/test_config.py

# Run a specific test
pytest deepref/tests/test_config.py::test_should_return_correct_preprocessing_combinations_while_combining
```

### Training
```bash
# Train with default dataset (semeval2010), reads hyperparams from deepref/hyperparameters/hyperparams_<dataset>.json
python deepref/framework/train.py --dataset semeval2010
```

### Hyperparameter Optimization
```bash
python deepref/optimization/bo_optimizer.py -d semeval2010 -m micro_f1 -t 50
# -d: dataset (semeval2010, semeval20181-1, semeval20181-2, ddi)
# -m: metric (micro_f1, macro_f1, acc)
# -t: number of Optuna trials
```

### Dataset Download & Preprocessing
```bash
bash benchmark/download_<dataset_name>.sh
# Optionally configure NLP tool before preprocessing:
export NLP_TOOL=spacy  # or stanza
export NLP_MODEL=en_core_web_sm
```

## Architecture

DeepREF is a framework for deep learning-based relation classification (sentence-level NLP task). The codebase was initially based on OpenNRE and adds optimization, preprocessing, and fine-tuning capabilities.

### Data Flow
1. **Raw datasets** → downloaded via `benchmark/download_<dataset>.sh` scripts
2. **Converters** (`deepref/dataset/converters/`) → transform raw data into standard CSV format stored in `benchmark/<dataset>/original/`
3. **Preprocessors** (`deepref/dataset/preprocessors/`) → apply text transformations and write `.txt` files to `benchmark/<dataset>/<preprocessing_str>/`
4. **Training** (`deepref/framework/train.py`) → loads `.txt` files, builds encoder + model, trains and evaluates

### Key Modules

**`deepref/config.py`** — Central configuration: lists all valid `DATASETS`, `MODELS`, `METRICS`, `PREPROCESSING_TYPES`, `PRETRAIN_WEIGHTS`, and default `HPARAMS`. Add new datasets/models here.

**`deepref/framework/`**
- `train.py` — `Training` class orchestrates the full pipeline: preprocessing, encoder selection, model creation, training loop
- `sentence_re.py` / `sentence_re_trainer.py` — Core training/eval framework wrapping PyTorch
- `fine_tuner.py` — Alternative LoRA-based fine-tuning path using HuggingFace `Trainer` + PEFT
- `data_loader.py` — `SentenceREDataset` / `SentenceRELoader` for the `.txt`-format data; `BagREDataset` for bag-level RE

**`deepref/encoder/`**
- `sentence_encoder.py` — `SentenceEncoder`: generic HuggingFace `AutoModel` wrapper with entity special tokens (`<e1>`, `</e1>`, `<e2>`, `</e2>`) and average pooling
- `bert_encoder.py` — `BERTEncoder` (CLS token) and `BERTEntityEncoder` (entity marker positions)
- `base_bert_encoder.py` — `EBEMEncoder` with optional position/POS/dependency embeddings

**`deepref/model/`**
- `softmax_nn.py` — `SoftmaxNN`: wraps an encoder + linear classifier head (primary model)
- `softmax_mlp.py` — MLP variant
- `pairwise_ranking_loss.py` — Custom loss for CRCNN model

**`deepref/dataset/`**
- `re_dataset.py` — `REDataset`: PyTorch Dataset reading from CSV (used by `FineTuner`)
- `dataset.py` — `Dataset`: manages `Sentence` objects and writes to `.txt`/CSV formats
- `sentence.py` — `Sentence`: holds token, entity, POS, dependency, NER, and relation data
- `converters/` — One converter per dataset, inheriting `DatasetConverter`; must implement `get_entity_dict()` and `get_sentences()` (generator with `yield`)
- `preprocessors/` — One preprocessor per text transformation type (`sw`, `p`, `b`, `d`, `eb`, `nb`)

**`deepref/optimization/`**
- `bo_optimizer.py` — Bayesian optimization via Optuna (TPE sampler + Hyperband pruner); saves result visualizations to `results/<dataset>/`

### Hyperparameter Files
`deepref/hyperparameters/hyperparams_<dataset>.json` — JSON file specifying model, pretrain weights, batch size, lr, preprocessing list, and embedding flags for a given dataset. Auto-created with defaults if missing.

### Adding New Components

**New model**: Create encoder in `deepref/encoder/`, register in `deepref/encoder/__init__.py`, add name to `MODELS` in `deepref/config.py`, add instantiation logic in `deepref/framework/train.py`.

**New dataset**: Add download script `benchmark/download_<dataset>.sh`, create converter in `deepref/dataset/converters/` inheriting `DatasetConverter`, add dataset name to `DATASETS` in `deepref/config.py`.

### Data Formats
- **CSV** (intermediate): columns `original_sentence, e1, e2, relation_type, pos_tags, dependencies_labels, ner, sk_entities`; tab-separated; stored in `benchmark/<dataset>/original/`
- **TXT** (training input): each line is a Python dict string with keys `token, h, t, relation, pos_tags, deps, ner, sk`; stored in `benchmark/<dataset>/<preprocessing_str>/`
- **rel2id JSON**: `benchmark/<dataset>/<dataset>_rel2id.json` — maps relation name → integer id

### Results
Training results written to `results/<dataset>/ResultsDeepREF_<dataset>_<datetime>.txt`. Model checkpoints saved to `ckpt/<dataset>_<model>.pth.tar`.
