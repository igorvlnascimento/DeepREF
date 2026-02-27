import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class SentenceEncoder(nn.Module):
    def __init__(self,
                 model_name,
                 max_length=512,
                 padding_side="left",
                 device="cpu",
                 attn_implementation="eager",
                 trainable=False):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation=attn_implementation,
        ).to(device)

        if not trainable:
            self.freeze_model()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]
        })
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.max_length = max_length
        self.padding_side = padding_side

    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Average pooling with attention mask."""
        last_hidden_states = last_hidden_states.to(torch.float32)
        last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        return F.normalize(embedding, dim=-1)

    def tokenize(self, item):
        return self.tokenizer(
            item,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def forward(self, texts: str | list[str]) -> torch.Tensor:
        """Encode one or more texts and return L2-normalised average-pool embeddings.

        Args:
            texts: a single string or a list of strings to encode.
        Returns:
            Float32 tensor of shape (N, hidden_dim), L2-normalised.
        """
        device = next(self.model.parameters()).device
        batch = self.tokenize(texts)
        batch = {k: v.to(device) for k, v in batch.items()}
        model_outputs = self.model(**batch)
        return self.average_pool(model_outputs.last_hidden_state, batch["attention_mask"])

    def freeze_model(self):
        for p in self.model.parameters():
            p.requires_grad = False
