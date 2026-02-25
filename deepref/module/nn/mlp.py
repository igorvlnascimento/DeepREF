import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, 
                 sentence_encoder,
                 dropout=0,
                 num_layers=3,
                 activation_function=nn.ReLU()):
        """
        Args:
            input_size: dimention of input embedding
            kernel_size: kernel_size for CNN
            padding: padding for CNN
            hidden_size: hidden size
        """
        super().__init__()
        hidden_size = sentence_encoder.model.config.hidden_size
        self.num_layers = num_layers

        layers = []
        for i in range(self.num_layers):
            layers.append(nn.Linear(hidden_size//2**i, hidden_size//2**(i+1)))
            layers.append(activation_function)
            layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            input features
        Return:
            output features: (B, H_EMBED)
        """
        x = self.mlp(x)
        return x
