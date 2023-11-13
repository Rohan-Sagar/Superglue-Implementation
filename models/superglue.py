import torch
import torch.nn as nn

class KeypointsMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KeypointsMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)
