import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        loss = 0
        y_pred += 1e-7
        for i in range(len(y_true)):
            loss += -1 * (y_true[i] * np.log(y_pred[i]) + (1-y_true[i]) * np.log(1-y_pred[i]))

        loss = loss / len(y_true)
        return round(loss, 4)
        pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        y_pred += 1e-7
        loss = 0
        # m * x
        n = len(y_true[0])
        m = len(y_true)
        counter = 0
        for i in range(m):
            for j in range(n):
                if y_true[i][j] == 1:
                    counter += 1
                    loss += -1 * (np.log(y_pred[i][j]))

        loss = loss / counter
        return round(loss, 4)  
        pass
