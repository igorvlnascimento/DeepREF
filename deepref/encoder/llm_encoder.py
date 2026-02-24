import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class LLMEncoder(nn.Module):
    def __init__(self, model_name, max_length=512, padding_side="left", device="cpu", attn_implementation="eager"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name,
                                               trust_remote_code=True,
                                               torch_dtype=torch.float16,
                                               attn_implementation=attn_implementation
                                               ).eval()
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.padding_side = padding_side


    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Average pooling with attention mask."""

        last_hidden_states = last_hidden_states.to(torch.float32)
        last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        embedding = F.normalize(embedding, dim=-1)
        
        return embedding


    def forward(self, input_texts):
        batch_dict = self.tokenizer(input_texts, 
                                    max_length=self.max_length, 
                                    padding=True, 
                                    truncation=True, 
                                    return_tensors="pt")
        attention_mask = batch_dict["attention_mask"]
        model_outputs = self.model(**batch_dict)

        return self.average_pool(model_outputs.last_hidden_state, attention_mask)