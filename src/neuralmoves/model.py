from __future__ import annotations
import torch
from torch import nn


class Net(nn.Module):
    """
    Lightweight MLP used for NeuralMOVES submodels.
    Input:  [v, a, grade, temp_C, humid_pct]  -> shape (N, 5)
    Output: running-exhaust CO2 rate in g/s (pre-floor)
    """
    def __init__(self, in_dim: int = 5, hidden: tuple[int, int] = (5, 5)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
