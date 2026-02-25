import torch
import torch.nn as nn
import torch.nn.functional as F

class CRCNN(nn.Module):

    def __init__(self, 
                 input_size=50, 
                 max_length=128, 
                 hidden_size=230, 
                 kernel_size=3, 
                 padding_size=1,
                 dropout=0,
                 activation_function=torch.tanh):
        super().__init__()
        # Hyperparameters
        self.drop = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function

        self.conv = nn.Conv1d(input_size, hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(max_length)


    def forward(self, x):
        """
        Args:
            input features: (B, L, I_EMBED)
        Return:
            output features: (B, H_EMBED)
        """
        x = x.transpose(1, 2) # (B, EMBED, L)
        x = self.act(self.conv(x)) # (B, H, L)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return x

