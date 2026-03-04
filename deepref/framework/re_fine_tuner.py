from typing import Any, Dict

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from deepref.dataset.re_dataset import REDataset


class REFineTuner:
    """Fine-tunes a HuggingFace sequence-classification model for relation extraction.

    Supports two modes:

    * **LoRA** (``lora_config_parameters`` is not ``None``): wraps the base model
      with a PEFT LoRA adapter.  Only the adapter weights are trained; the
      backbone is frozen.
    * **Full fine-tuning** (``lora_config_parameters`` is ``None``): all model
      parameters are updated during training.

    Args:
        model_name: HuggingFace model identifier.
        train_dataset: training :class:`~deepref.dataset.re_dataset.REDataset`.
        test_dataset: evaluation :class:`~deepref.dataset.re_dataset.REDataset`.
        training_parameters: keyword arguments forwarded to
            :class:`~transformers.TrainingArguments` (must include
            ``"output_dir"``).
        lora_config_parameters: keyword arguments forwarded to
            :class:`~peft.LoraConfig`, or ``None`` to perform full fine-tuning.
    """

    def __init__(
        self,
        model_name: str,
        train_dataset: REDataset,
        test_dataset: REDataset,
        n_new: int,
        training_parameters: Dict[str, Any],
        lora_config_parameters: Dict[str, Any] | None = None,
    ) -> None:

        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(train_dataset.rel2id),
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        base_model.resize_token_embeddings(len(train_dataset.tokenizer))

        with torch.no_grad():
            old_embeddings = base_model.get_input_embeddings().weight[:-n_new]
            avg_embedding = old_embeddings.mean(dim=0)
            base_model.get_input_embeddings().weight[-n_new:] = avg_embedding

        if lora_config_parameters is not None:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(**lora_config_parameters)
            self.model = get_peft_model(base_model, lora_config)
            self._use_lora = True
        else:
            self.model = base_model
            self._use_lora = False

        training_args = TrainingArguments(**training_parameters)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
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
        self.model.save_pretrained(self.save_model_name)