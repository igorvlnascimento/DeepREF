import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module.nn import LSTM
from .base_encoder import BaseEncoder

class LSTMEncoder(BaseEncoder):

    def __init__(self, 
                 token2id, 
                 max_length=128, 
                 hidden_size=256, 
                 word_size=50,
                 position_size=5,
                 blank_padding=True,
                 word2vec=None,
                 bidirectional=False,
                 dropout=0,
                 activation_function=F.relu,
                 mask_entity=False):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            hidden_size: hidden size
            word_size: size of word embedding
            position_size: size of position embedding
            blank_padding: padding for LSTM
            word2vec: pretrained word2vec numpy
            bidirectional: if LSTM is bidirectional or not
        """
        # Hyperparameters
        super(LSTMEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, blank_padding, word2vec, mask_entity=mask_entity)
        input_size = word2vec.shape[-1] + 2 * position_size
        self.drop = nn.Dropout(dropout)
        self.lstm = LSTM(input_size=input_size, dropout=dropout, bidirectional=bidirectional, hidden_size=hidden_size)
        self.pool = nn.MaxPool1d(self.max_length)
        self.act = activation_function


    def forward(self, token, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            pos1: (B, L), relative position to head entity
            pos2: (B, L), relative position to tail entity
        Return:
            (B, EMBED), representations for sentences
        """
        x = torch.cat([self.word_embedding(token), 
                       self.pos1_embedding(pos1), 
                       self.pos2_embedding(pos2)], 2) # (B, L, EMBED)
        x = self.act(self.lstm(x))
        x = x.transpose(1, 2) # (B, EMBED, L)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        
        return x

    def tokenize(self, item):
        return super().tokenize(item)
