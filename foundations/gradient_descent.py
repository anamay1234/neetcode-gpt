class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        for i in range(iterations):
            # how much to change our parameter by
            adjustParameter = learning_rate * (2*init)

            # adjusting our parameter
            init = init - adjustParameter

        return round(init, 5)


