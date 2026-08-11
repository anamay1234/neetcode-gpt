import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        n = X.shape[0]
        # get the shape of what the weights should be and initalize to be 0s
        w = np.zeros(X.shape[1])
        b = 0

        for _ in range(epochs):
            # predictions 
            y_hat = X @ w + b

            # loss
            MSE = (1 / n) * np.sum((y_hat - y) ** 2)

            # finding the gradients with respect to w and b
            grad_w = (2 / n) * X.T @ (y_hat - y)
            grad_b = (2 / n) * np.sum(y_hat - y)

            # updating w and b
            w = w - lr * (grad_w)
            b = b - lr * (grad_b)

        return np.round(w, 5), round(b, 5)
