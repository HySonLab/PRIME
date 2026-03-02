import torch
from torch import nn

class MLP_Head(nn.Module):
    def __init__(self, in_dim, out_dim,
                    hidden_dim=512,
                    num_layers=3,
                    dropout=0.3):
            super().__init__()

            layers = []
            dim = in_dim

            for _ in range(num_layers - 1):
                layers.append(nn.Linear(dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                dim = hidden_dim

            layers.append(nn.Linear(dim, out_dim))

            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)