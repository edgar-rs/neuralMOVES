"""Neural network architecture for NeuralMOVES.

This architecture MUST match the trained weights in NN_3/*.pt files.
Do not modify without retraining all models.
"""

from __future__ import annotations
import torch
from torch import nn


class Net(nn.Module):
    """
    Lightweight MLP for NeuralMOVES running emissions prediction.
    
    Architecture: 5 inputs → 64 hidden → 64 hidden → 1 output
    Activation: tanh
    
    Input features (in order):
        1. Temperature (°C)
        2. Humidity (%)
        3. Speed (m/s)
        4. Acceleration (m/s²)
        5. Grade (%)
        
    Output: Running-exhaust CO2 rate (g/s) before idling floor
    """
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 1):
        """
        Initialize the neural network.
        
        Parameters
        ----------
        input_dim : int
            Number of input features (default: 5)
        hidden_dim : int
            Number of neurons in each hidden layer (default: 64)
        output_dim : int
            Number of outputs (default: 1)
        """
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.tanh1 = nn.Tanh()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh2 = nn.Tanh()
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, 5) where N is batch size
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, 1) with predicted emissions
        """
        x = self.tanh1(self.layer1(x))
        x = self.tanh2(self.layer2(x))
        return self.layer3(x)
