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
                 activation_function=nn.ReLU()):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.model = MLP(self.sentence_encoder, 
                         dropout=dropout, 
                         num_layers=num_layers, 
                         activation_function=activation_function)
        input_dim = self.sentence_encoder.model.config.hidden_size // 2**num_layers
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
