import torch.nn as nn

class MLP(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 dropout=0,
                 num_layers=3,
                 activation_function=nn.ReLU()):
        """
        Args:
            hidden_size: input embedding dimension
        """
        super().__init__()
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
