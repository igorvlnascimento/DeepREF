import torch.nn as nn
import torch.nn.functional as F

from deepref.module.nn import LSTM

class SentenceLSTM(nn.Module):

    def __init__(self, 
                 input_size=50, 
                 max_length=128, 
                 hidden_size=256, 
                 bidirectional=False,
                 dropout=0,
                 activation_function=F.relu):
        # Hyperparameters
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.lstm = LSTM(input_size=input_size, 
                         dropout=dropout, 
                         bidirectional=bidirectional, 
                         hidden_size=hidden_size, 
                         num_layers=2)
        self.pool = nn.MaxPool1d(max_length)
        self.act = activation_function

    def forward(self, x):
        x = self.act(self.lstm(x))
        print(x.shape)
        x = self.pool(x).squeeze(-1)
        x = x.view(-1).unsqueeze(0)
        x = self.drop(x)
        
        return x