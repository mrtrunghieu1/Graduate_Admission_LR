import numpy as np


def features_normalize(X):
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    mu = np.mean(X_norm, axis=0)
    for i in range(X.shape[1]):
        X_norm[:, i] -= mu[i]

    sigma = np.zeros((1, X.shape[1]))
    sigma = np.std(X_norm, axis=0)
    for i in range(X.shape[1]):
        X_norm[:, i] /= sigma[i]

    return X_norm, mu, sigma