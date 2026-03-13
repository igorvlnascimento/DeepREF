from torch import nn
import torch
import torch.nn.functional as F

from deepref.module.nn.mlp import MLP
from .base_model import SentenceRE

class CombineREClassifier(SentenceRE):
    """
    Softmax MLP classifier for sentence-level relation extraction.
    """

    def __init__(self,
                 sentence_encoder,
                 num_class,
                 rel2id,
                 activation_function=nn.GELU()):
        """
        Args:
            sentence_encoder: encoder for sentences (may be None when hidden_size is given)
            num_class: number of classes
            rel2id: dictionary of relation name -> id mapping
            hidden_size: input embedding dimension; inferred from sentence_encoder when None
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.h1 = sentence_encoder.h1
        self.h2 = sentence_encoder.h2
        self.num_class = num_class
        self.act = activation_function
        self.softmax = nn.Softmax(-1)
        self.model_bert = nn.Sequential(
            nn.LayerNorm(self.h1, elementwise_affine=True),
            nn.Linear(self.h1, 1152),
            self.act,
            nn.Dropout(0.2),
        )

        self.model_qwen = nn.Sequential(
            nn.LayerNorm(self.h2, elementwise_affine=False),
            nn.Linear(self.h2, 128),
            self.act,
            nn.Dropout(0.4),
        )

        self.model_final = nn.Sequential(
            nn.Linear(1280, 640),
            self.act,
            nn.LayerNorm(640, elementwise_affine=True),
            nn.Dropout(0.3),
            nn.Linear(640, num_class),
        )
        self.rel2id = rel2id
        self.id2rel = {}
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score
    
    def forward(self, **args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(**args) # (B, H)
        rep1 = rep[..., :self.h1]
        rep2 = rep[..., self.h1:]
        hidden1 = self.model_bert(rep1) # (B, H)
        hidden2 = self.model_qwen(rep2) # (B, H)
        hidden = torch.cat([hidden1, hidden2], dim=1)
        logits = self.model_final(hidden) # (B, N)
        return logits
