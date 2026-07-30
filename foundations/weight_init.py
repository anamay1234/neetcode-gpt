import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2/(fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2/fan_in)
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.

        torch.manual_seed(0)

        # Store dimensions of each layer
        # Example: input=64, hidden=64, layers=5
        # dims = [64,64,64,64,64,64]
        dims = [input_dim] + [hidden_dim] * num_layers

        weights = []

        # Create weights for every layer
        for i in range(num_layers):

            # Xavier initialization
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))

            # Kaiming initialization for ReLU
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / dims[i])

            # Normal random initialization
            else:
                std = 1.0

            # Weight shape = (fan_out, fan_in)
            w = torch.randn(dims[i + 1], dims[i]) * std
            weights.append(w)

        # w = 5 matrixes of size 64*64 with the weights being initalized with their std

        # Create random input
        # Shape: (batch_size, input_dim)
        x = torch.randn(1, input_dim)

        # Store standard deviation after each layer
        stds = []

        # Forward pass through all layers
        for w in weights:

            # Linear layer:
            # input * weights
            x = x @ w.T

            # Apply ReLU activation
            x = torch.relu(x)

            # Record activation size
            stds.append(round(x.std().item(), 2))

        return stds