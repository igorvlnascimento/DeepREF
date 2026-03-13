from torch import nn
import torch.nn.functional as F

from deepref.module.nn.mlp import MLP
from .base_model import SentenceRE

class SoftmaxMLP(SentenceRE):
    """
    Softmax MLP classifier for sentence-level relation extraction.
    """

    def __init__(self,
                 sentence_encoder,
                 num_class,
                 rel2id,
                 dropout=0,
                 num_layers=3,
                 activation_function=nn.ReLU(),
                 hidden_size: int | None = None):
        """
        Args:
            sentence_encoder: encoder for sentences (may be None when hidden_size is given)
            num_class: number of classes
            rel2id: dictionary of relation name -> id mapping
            hidden_size: input embedding dimension; inferred from sentence_encoder when None
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        if hidden_size is None:
            hidden_size = sentence_encoder.model.config.hidden_size
        self.model = MLP(hidden_size,
                         dropout=dropout,
                         num_layers=num_layers,
                         activation_function=activation_function)
        input_dim = hidden_size // 2**num_layers
        self.fc = nn.Linear(input_dim, num_class)
        self.softmax = nn.Softmax(-1)
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
    
    def forward_from_emb(self, emb):
        """Forward from a pre-computed embedding tensor, bypassing sentence_encoder."""
        return self.fc(self.model(emb))

    def forward(self, **args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(**args) # (B, H)
        logits = self.model(rep) # (B, H)
        logits = self.fc(logits) # (B, N)
        return logits
