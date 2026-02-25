import torch
import torch.nn as nn
import torch.nn.functional as F

class SentencePCNN(nn.Module):

    def __init__(self, 
                 input_size=50, 
                 max_length=128,
                 hidden_size=230,
                 kernel_size=3, 
                 padding_size=1,
                 dropout=0.0,
                 activation_function=F.relu):
        # hyperparameters
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function
    
        self.conv = nn.Conv1d(input_size, hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(max_length)
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.mask_embedding.weight.requires_grad = False
        self._minus = -100

        #self.hidden_size *= 3

    def forward(self, x, mask):
        """
        Args:
            x: (B, L), index of tokens
        Return:
            (B, EMBED), representations for sentences
        """
        x = x if self.sentence_classifier else x.transpose(1, 2) # (B, EMBED, L)
        x = self.conv(x) # (B, H, L)

        mask = self.mask_embedding(mask) if self.sentence_classifier else self.mask_embedding(mask).transpose(1, 2) # (B, L)
        mask = 1 - mask # (B, L) -> (B, L, 3) -> (B, 3, L)
        pool1 = self.pool(self.act(x + self._minus * mask[:, 0:1])) 
        pool2 = self.pool(self.act(x + self._minus * mask[:, 1:2]))
        pool3 = self.pool(self.act(x + self._minus * mask[:, 2:3]))
        x = torch.cat([pool1, pool2, pool3], 1) # (B, 3H, 1)
        x = x.squeeze(2) # (B, 3H)
        x = self.drop(x)

        return x