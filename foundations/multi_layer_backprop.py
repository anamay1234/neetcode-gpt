import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # Forward Pass

        # Layer 1 - calculating the z's
        z1 = np.dot(x, W1.T) + b1
        # Layer 1 - z's go thru relu
        a1 = np.maximum(0, z1)
        # Layer 2 - calculating the 2nd layer z's
        z2 = np.dot(a1, W2.T) + b2
        # Calculating Error!
        loss = np.sum((z2 - y_true) ** 2) / len(z2)

        # Backward Pass

        # Lets find dl/dw_2 = dl/dz_2 * dz_2/dw_2
        # dl/dz_2:

        # Loss = (1/N)(z2 - y_true)^2
        #
        # Derivative:
        # dL/dz2 = 2 * (z2 - y_true) / N
        dL_dz2 = 2 * (z2 - y_true) / len(z2)

        # dz_2/dw_2:

        # z_2 = a_1*W_2 + b_2
        #
        # Derivative:
        # a_1
        dz_2_dw_2 = a1

        # Thus dl/dw_2 = dl/dz_2 * dz_2/dw_2
        dl_dw2 = np.outer(dL_dz2, a1)

        # Lets find d1/db_2 = dl/dz_2 * dz_2/db_2

        # dz_2/db_2:
        # z_2 = a_1*W_2 + b_2
        #
        # Derivative:
        # 1
        dL_db2 = dL_dz2 * 1

        # Now lets find dl/dw1 = dl/dz_2 * dz_2/da_1 * da_1/dz_1 * dz_1/dw_1
        # we already have dl/dz_2

        # lets now find dz_2/da_1:
        # z_2 = a_1*W_2 + b_2
        #
        # Derivative:
        # W_2
        dz2_da1 = W2

        # lets now find da_1/dz_1:
        # a1 = np.maximum(0, z1)
        #
        # ReLU derivative:
        # if z1 > 0 -> 1
        # if z1 <= 0 -> 0
        da1_dz1 = (z1 > 0)

        # lets now find dz_1/dw_1:
        # z_1 = X*W_1 + b_1
        #
        # Derivative:
        # X
        dz1_dw1 = x

        # lets now find dz_1/db_1:
        dz1_db1 = 1

        # Now lets get dL/dw1 = dL_dz2 * dz2_da1 * da1_dz1 * dz1_dw1

        # Move gradient through W2:
        # dL/da1 = dL/dz2 * dz2/da1
        dL_da1 = dL_dz2 @ dz2_da1

        # Move gradient through ReLU:
        # dL/dz1 = dL/da1 * da1/dz1
        dL_dz1 = dL_da1 * da1_dz1

        # Get dL/dW1:
        dL_dW1 = np.outer(dL_dz1, x)

        # Get dL/db1:
        # dz1/db1 = 1
        # therefore dL/db1 = dz1
        dL_db1 = dL_dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dL_dW1, 4).tolist(),
            "db1": np.round(dL_db1, 4).tolist(),
            "dW2": np.round(dl_dw2, 4).tolist(),
            "db2": np.round(dL_db2, 4).tolist()
        }





        




        

