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

        # Step 1:
        # Forward pass
        #
        # Calculate the raw neuron output:
        # z = x · w + b
        #
        # Then apply sigmoid activation:
        # y_hat = sigmoid(z)
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-1 * z))

        # Calculate loss:
        # This measures how far the prediction is from the true value.
        loss = 0.5 * (y_hat - y_true)**2


        # Step 2:
        # Backpropagation
        #
        # We move backwards through the same path: 
        #   take derivative of loss w.r.t to y_hat, with respect to sigmoid, ect
        #
        # Loss → y_hat → z → weights
        #
        # Each derivative tells us how much each part
        # contributed to the final error.


        # Step 3:
        # Take derivative of loss w.r.t to y_hat: 
        # derivative of Loss = 0.5 * (y_hat - y_true)**2 is -> y_hat - y_true
        # Now to take derivative of loss w.r.t to y_hat we plug in our numbers into this above
        # derivative 
        dL_dŷ = y_hat - y_true

        # Step 4:
        # Pass through sigmoid derivative:
        #
        # dŷ/dz = y_hat * (1 - y_hat)
        #
        # This tells us how much the sigmoid output
        # changes when z changes.
        dŷ_dz = y_hat * (1 - y_hat)

        # Step 5:
        # Chain rule:
        #
        # dL/dz = dL/dŷ * dŷ/dz
        #
        # This combines the effect of the loss
        # and the sigmoid activation.
        dL_dz = dL_dŷ * dŷ_dz
        dL_dz = dL_dŷ * dŷ_dz

        # Step 6:
        # Calculate weight gradients:
        #
        # dL/dw = dL/dz * dz/dw
        #
        # Since:
        # z = x1*w1 + x2*w2 + ... + b
        #
        # dz/dw = x
        dL_dw = dL_dz * x

        # Step 7:
        # Calculate bias gradient:
        #
        # dz/db = 1
        #
        # Therefore:
        # dL/db = dL/dz
        dL_db = dL_dz

        return np.round(dL_dw, 5), round(dL_db, 5)