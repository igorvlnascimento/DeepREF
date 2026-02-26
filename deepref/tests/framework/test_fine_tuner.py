import os

from peft import TaskType
import pytest
import inspect

from transformers import TrainingArguments

from deepref.dataset.re_dataset import REDataset
from deepref.encoder.sentence_encoder import SentenceEncoder
from deepref.framework.fine_tuner import FineTuner

MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"

@pytest.fixture(scope="module")
def datasets():
    encoder = SentenceEncoder(MODEL_NAME)
    train_dataset = REDataset("benchmark/semeval2010", encoder.tokenizer, dataset_split="train")
    train_dataset.df = train_dataset.df[:3]
    test_dataset = REDataset("benchmark/semeval2010", encoder.tokenizer, dataset_split="test")
    test_dataset.df = test_dataset.df[:3]
    return train_dataset, test_dataset

def test_fine_tuner(datasets):
    train_dataset, test_dataset = datasets
    lora_config = {
        "task_type": TaskType.SEQ_CLS,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"]
    }
    train_parameters = {
        "output_dir": "./smollm-relation-extration-lora",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 4,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "report_to": []
    }

    valid_keys = inspect.signature(TrainingArguments).parameters
    filtered_args = {
        k: v for k, v in train_parameters.items() if k in valid_keys
    }

    finetuner = FineTuner(MODEL_NAME, train_dataset, test_dataset, filtered_args, lora_config)

    finetuner.finetune_model()
    finetuner.save_model()

    assert os.path.exists("./smollm-relation-extration-lora/")