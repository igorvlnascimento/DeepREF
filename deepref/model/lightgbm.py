from lightgbm import LGBMClassifier
from torch import nn

from .base_model import SentenceRE

class XGBoostClassifier(SentenceRE):
    """
    Softmax MLP classifier for sentence-level relation extraction.
    """

    def __init__(self, 
                 sentence_encoder, 
                 rel2id,
                 n_estimators=100,
                 learning_rate=0.05):
        """
        Args:
            sentence_encoder: encoder for sentences
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.gbdt_clf = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, use_label_encoder=False)
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
        logits = self.gbdt_clf.predict_proba(rep) # (B, H)
        return logits
