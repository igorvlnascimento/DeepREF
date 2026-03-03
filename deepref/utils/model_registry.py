from typing import Dict

import torch
from transformers import AutoModel, AutoTokenizer

class ModelRegistry:
    _instance = None
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, model_name: str, device="cuda", attn_implementation="eager", trainable=False):
        if model_name not in self._models:
            print(f"Loading {model_name} onto {device}...")
            # Use float32 for trainable models: float16 gradients overflow
            # to inf/NaN during the backward pass without gradient scaling.
            # float16 is kept for frozen models (inference only) to save memory.
            dtype = torch.float32 if trainable else torch.float16
            self._models[model_name] = {
                "model": AutoModel.from_pretrained(model_name,
                                                   trust_remote_code=True,
                                                   torch_dtype=dtype,
                                                   attn_implementation=attn_implementation).to(device),
                "tokenizer": AutoTokenizer.from_pretrained(model_name),
                "device": device,
                "trainable": trainable,
            }
            n_new = self._models[model_name]["tokenizer"].add_special_tokens({
                "additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]
            })
            self._models[model_name]["model"].resize_token_embeddings(len(self._models[model_name]["tokenizer"]))
            with torch.no_grad():
                old_embeddings = self._models[model_name]["model"].get_input_embeddings().weight[:-n_new]
                avg_embedding = old_embeddings.mean(dim=0)
                self._models[model_name]["model"].get_input_embeddings().weight[-n_new:] = avg_embedding
            if not trainable:
                self.freeze_model(model_name)
            print(f"✅ {model_name} loaded onto {device} (trainable={trainable})")
        return self._models[model_name]

    def is_loaded(self, model_name: str) -> bool:
        return model_name in self._models

    def unload(self, model_name: str):
        if model_name in self._models:
            del self._models[model_name]
            torch.cuda.empty_cache()
            print(f"🗑️ {model_name} unloaded from GPU")

    def list_loaded(self):
        return list(self._models.keys())

    def tokenize(self, model_name: str, text: str | list[str], **kwargs):
        """Tokenizes text and returns input_ids already on the correct device."""
        entry = self._get_or_raise(model_name)
        tokenizer = entry["tokenizer"]
        device = entry["device"]

        kwargs.setdefault("padding", True)
        kwargs.setdefault("truncation", True)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            **kwargs
        )
        # Move all tensors to the same device as the model
        return {k: v.to(device) for k, v in inputs.items()}
    
    def tokenize_as_str(self, model_name: str, text: str | list[str], **kwargs):
        """Tokenizes text and returns input_ids already on the correct device."""
        entry = self._get_or_raise(model_name)
        tokenizer = entry["tokenizer"]

        return tokenizer.tokenize(
            text
        )
    
    def convert_tokens_to_ids(self, model_name: str, tokens: list[str]):
        return self._get_or_raise(model_name)["tokenizer"].convert_tokens_to_ids(tokens)
    
    def get_tokenizer_mask_token(self, model_name: str):
        return self._get_or_raise(model_name)["tokenizer"].mask_token
    
    def get_tokenizer_cls_token(self, model_name: str):
        return self._get_or_raise(model_name)["tokenizer"].cls_token
    
    def get_tokenizer_sep_token(self, model_name: str):
        return self._get_or_raise(model_name)["tokenizer"].sep_token
    
    def get_tokenizer_pad_token_id(self, model_name: str):
        return self._get_or_raise(model_name)["tokenizer"].pad_token_id

    def run(self, model_name: str, text: str | list[str], **kwargs):
        """Tokenizes and runs a full forward pass. Returns model outputs."""
        entry = self._get_or_raise(model_name)
        model = entry["model"]
        trainable = entry["trainable"]

        inputs = self.tokenize(model_name, text, **kwargs)

        if trainable:
            outputs = model(**inputs)
        else:
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)

        return outputs, inputs

    def run_from_input_ids(self, model_name: str, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """Runs inference directly from pre-built input_ids tensor."""
        entry = self._get_or_raise(model_name)
        model = entry["model"]
        device = entry["device"]
        trainable = entry["trainable"]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        if trainable:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs

    def _get_or_raise(self, model_name: str):
        if model_name not in self._models:
            raise RuntimeError(f"Model '{model_name}' is not loaded. Call .load() first.")
        return self._models[model_name]
    
    def freeze_model(self, model_name: str):
        """Freeze all model parameters."""
        for p in self._models[model_name]["model"].parameters():
            p.requires_grad = False

    def set_train_mode(self, model_name: str, mode: bool) -> None:
        """Set the backbone's train/eval mode, but only when trainable=True.

        Frozen models are always kept in eval mode so that dropout and batch
        norm behave correctly during inference.
        """
        entry = self._get_or_raise(model_name)
        if entry["trainable"]:
            entry["model"].train(mode)

    def get_model_hidden_size(self, model_name: str):
        return self._models[model_name]["model"].config.hidden_size
    
    def get_model_device(self, model_name: str):
        return next(self._models[model_name]["model"].parameters()).device