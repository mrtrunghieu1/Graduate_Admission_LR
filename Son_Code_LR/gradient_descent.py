import numpy as np
from cost_function import cost_function


def gradient_descent(X, y, W, alpha, num_iters):
    """
    X: m x n
    y: m x 1
    W: n x 1
    alpha: learning rate
    num_iters: number of iterations
    """
    m = y.shape[0] # number of tranining examples
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        delta = np.zeros((X.shape[1], 1))
        for j in range(m):
            delta += np.dot(W.T, X[j:(j+1), :].T)[0] - y[j] * X[j:(j+1), :].T
        delta /= m
        W -= alpha * delta

        # Save the cost J in every iteration
        J_history[i] = cost_function(X, y, W)
    
    return W, J_history