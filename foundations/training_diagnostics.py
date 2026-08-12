import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []

        inpuT = x
        for layer in model:
            inpuT = layer(inpuT)

            # if the layer is a linear output
            if isinstance(layer, nn.Linear):

                dictionary = {}
                dictionary["mean"] = inpuT.mean().item()
                dictionary["std"] = inpuT.std().item()

                dead = (inpuT <= 0).all(dim=0)
                dictionary["dead_fraction"] = round(dead.float().mean().item(), 4)

                stats.append(dictionary)

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        
        # Clear old gradients
        model.zero_grad()

        stats = []

        # Forward pass
        y_hat = model(x)

        # Calculate MSE loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_hat, y)

        # Backward pass: calculate gradients
        loss.backward()

        # Inspect each Linear layer
        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad

                dictionary = {}
                dictionary["mean"] = round(grad.mean().item(), 4)
                dictionary["std"] = round(grad.std().item(), 4)
                dictionary["norm"] = round(torch.norm(grad).item(), 4)

                stats.append(dictionary)

        return stats


    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)

        # 1. Dead neurons
        for layer in activation_stats:
            if layer["dead_fraction"] > 0.5:
                return "dead_neurons"

        # 2. Exploding gradients
        for layer in gradient_stats:
            if layer["norm"] > 1000:
                return "exploding_gradients"

        # 3. Vanishing gradients
        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # 4. Activation std
        for layer in activation_stats:
            if layer["std"] < 0.1:
                return "vanishing_gradients"

            if layer["std"] > 10.0:
                return "exploding_gradients"

        # 5. Everything looks okay
        return "healthy"
