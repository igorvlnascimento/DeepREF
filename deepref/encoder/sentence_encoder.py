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
        self.model = AutoModel.from_pretrained(model_name,
                                               trust_remote_code=True,
                                               torch_dtype=torch.float16,
                                               attn_implementation=attn_implementation)
        self.model = self.model.to(device)
        if not trainable:
            self.frozen_model()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        new_tokens = ["<e1>", "</e1>", "<e2>", "</e2>"]
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": new_tokens
        })
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.max_length = max_length
        self.padding_side = padding_side


    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Average pooling with attention mask."""

        last_hidden_states = last_hidden_states.to(torch.float32)
        last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        embedding = F.normalize(embedding, dim=-1)
        
        return embedding
    
    def tokenize(self, item):
        text = item
        batch_dict = self.tokenizer(text, 
                                    max_length=self.max_length,
                                    padding="max_length", 
                                    truncation=True, 
                                    return_tensors="pt")
        return batch_dict


    def forward(self, **batch_dicts):
        model_outputs = self.model(**batch_dicts)
        attention_mask = batch_dicts["attention_mask"]

        return self.average_pool(model_outputs.last_hidden_state, attention_mask)

    def frozen_model(self):
        for p in self.model.parameters():
            p.requires_grad = False