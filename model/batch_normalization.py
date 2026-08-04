import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
                # Step 0: Convert everything to NumPy arrays
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)
        beta = np.array(beta, dtype=float)
        running_mean = np.array(running_mean, dtype=float)
        running_var = np.array(running_var, dtype=float)

        if training:

            # Step 1: Compute batch mean (column-wise)
            batch_mean = np.mean(x, axis=0)

            # Step 2: Compute batch variance (column-wise)
            batch_var = np.var(x, axis=0)

            # Step 3: Normalize
            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)

            # Step 4: Scale and shift
            y = gamma * x_hat + beta

            # Step 5: Update running statistics
            running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            running_var = (1 - momentum) * running_var + momentum * batch_var

        else:

            # Step 1: Normalize using running statistics
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)

            # Step 2: Scale and shift
            y = gamma * x_hat + beta

        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )
