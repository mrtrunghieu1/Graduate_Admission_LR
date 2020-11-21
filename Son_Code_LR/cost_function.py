import numpy as np


def cost_function(X, y, W):
    """
    X: (m x n) matrix of examples
    y: (m x 1) matrix of true values
    W: (n x 1) matrix of parameters
    """
    # vectorized implementation
    m = y.shape[0] # number of examples
    J = 0 # initial cost
    predictions = np.dot(X, W)
    square_error = np.dot((predictions - y).T, predictions - y) 
    J = np.sum(square_error) * (1/(2*m))
    
    return J

