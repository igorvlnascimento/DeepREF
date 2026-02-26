from typing import Any, Counter, Dict

import numpy as np
from peft import LoraConfig, get_peft_model
from sklearn.metrics import precision_recall_fscore_support
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from deepref.dataset.re_dataset import REDataset


class FineTuner:
    def __init__(self, 
                 model_name: str, 
                 train_dataset: REDataset, 
                 test_dataset: REDataset,
                 training_parameters: Dict[str, Any],
                 lora_config_parameters: Dict[str, Any]) -> None:

        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(train_dataset.rel2id),
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        lora_config = LoraConfig(
            **lora_config_parameters
        )

        self.peft_model = get_peft_model(base_model, lora_config)

        training_args = TrainingArguments(
            **training_parameters
        )

        self.trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics
        )

        self.save_model_name = training_parameters["output_dir"]


    def finetune_model(self):
        self.trainer.train()

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def save_model(self):
        self.peft_model.save_pretrained(f"./{self.save_model_name}")