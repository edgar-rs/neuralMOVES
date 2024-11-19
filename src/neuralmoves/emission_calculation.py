"""
This module provides functionality for computing vehicle emissions using a neural network model.

It includes a neural network class definition and a function to calculate emissions based on
various input parameters and driving cycle data.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class Net(nn.Module):
    """
    A neural network model for emissions prediction.

    This class defines a neural network with configurable input, hidden, and output dimensions.
    It uses tanh activation functions and is designed for emissions modeling tasks.

    Attributes:
        layer1 (nn.Linear): First linear layer
        tanh1 (nn.Tanh): First tanh activation
        layer2 (nn.Linear): Second linear layer
        tanh2 (nn.Tanh): Second tanh activation
        layer3 (nn.Linear): Output linear layer
    """

    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1):
        """
        Initialize the neural network.

        Args:
            input_dim (int): Number of input features (default: 5)
            hidden_dim (int): Number of neurons in hidden layers (default: 64)
            output_dim (int): Number of output values (default: 1)
        """
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.tanh1 = nn.Tanh()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh2 = nn.Tanh()
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Perform the forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output of the network
        """
        x = self.tanh1(self.layer1(x))
        x = self.tanh2(self.layer2(x))
        return self.layer3(x)

def get_emissions(source_type, model_year, fuel_type, temperature, humidity, df_speed_acceleration_grade):
    """
    Calculate emissions using a neural network model and idling emissions data.

    This function loads a pre-trained neural network model, processes input data,
    and computes emissions based on the model predictions and idling emissions.

    Args:
        source_type (int): Source type identifier
        model_year (int): Vehicle model year
        fuel_type (int): Fuel type identifier
        temperature (float): Ambient temperature
        humidity (float): Ambient humidity
        df_speed_acceleration_grade (pd.DataFrame): DataFrame containing speed, acceleration, and road grade data

    Returns:
        np.ndarray: Array of calculated emissions

    Raises:
        FileNotFoundError: If the idling emissions file or model file is not found
        ValueError: If input data is invalid or missing
    """
    # Load idling emissions data
    try:
        df_idling = pd.read_csv('idling_emissions.csv')
    except FileNotFoundError:
        raise FileNotFoundError("Idling emissions file not found.")

    # Get corresponding idling emission
    idling_emission = df_idling.loc[
        (df_idling['sourceTypeID'] == source_type) & 
        (df_idling['modelYear'] == model_year) & 
        (df_idling['fuelTypeID'] == fuel_type), 
        'emission_per_second_MOVES'
    ].values

    if len(idling_emission) == 0:
        raise ValueError("No matching idling emission data found.")

    idling_emission = idling_emission[0]

    # Load the neural network model
    model = Net()
    model_file_path = f"NN_3/NN_model_{model_year}_{source_type}_{fuel_type}.pt"
    try:
        model.load_state_dict(torch.load(model_file_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_file_path}")

    # Prepare input data
    df_input = df_speed_acceleration_grade.copy()
    df_input.insert(0, 'temperature', temperature)
    df_input.insert(1, 'humidity', humidity)

    # Convert input to tensor
    X = torch.tensor(df_input.values, dtype=torch.float32)

    # Get model predictions
    with torch.no_grad():
        nn_emission = model(X).numpy()

    # Calculate final emissions
    inst_emission = np.maximum(idling_emission, nn_emission)

    return inst_emission
