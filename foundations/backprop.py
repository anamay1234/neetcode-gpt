import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        pass
        # Forward Pass
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-1 * z))
        loss = 0.5 * (y_hat - y_true)**2

        # Backward Pass
        # Backprop just walks backward through that same path 
        # and uses derivatives to measure each part's contribution.
        # So we take derivative of loss w.r.t to y_hat
        # then derivative of above with respect to sigmoid
        # then derivative of above with respect to 

        # Take derivative of loss w.r.t to y_hat: 
        # derivative of Loss = 0.5 * (y_hat - y_true)**2 is -> y_hat - y_true
        # Now to take derivative of loss w.r.t to y_hat we plug in our numbers into this above
        # derivative 
        dL_dŷ = y_hat - y_true

        # Pass thru sigmoid
        # Sigmoid derivative
        dŷ_dz = y_hat * (1 - y_hat)

        # Chain rule: combine the two derivatives
        dL_dz = dL_dŷ * dŷ_dz

        # Weight gradients
        dL_dw = dL_dz * x

        # Bias gradient
        dL_db = dL_dz

        return np.round(dL_dw, 5), round(dL_db, 5)