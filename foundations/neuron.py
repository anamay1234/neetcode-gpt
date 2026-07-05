import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        y_hat = np.dot(x, w) + b
        if activation == "sigmoid":
            convert_y_hat_to_percentage = 1 / (1 + np.exp(-1 * y_hat))      
            return round(convert_y_hat_to_percentage, 5)   
        if activation == "relu":
            y_hat_with_relu = max(0.0, y_hat)
            return round(y_hat_with_relu, 5)
        


