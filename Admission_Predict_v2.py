import pandas as pd 
import numpy as np 

# read data
df = pd.read_csv("Admission_Predict.csv")
df = df.values
df = df[:, 1:]

# split train set
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
df = train_set.copy()
X = df[:, :-1]
y = df[:, -1].reshape(df.shape[0],1)

# normalize feature
def feature_normalization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    return X

one = np.ones((df.shape[0],1))
X = feature_normalization(X)
X = np.concatenate((one, X), axis=1)

# functions
def cost_function(X, y, weight):
    N = X.shape[0]
    cost = 0
    for i in range(N):
        cost = cost + (np.subtract(y[i], np.dot(X[i], weight)))**2
    return 1/(2*N) * cost

def gradient_descent(X, y, weight, learning_rate, iterations):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        grad = np.subtract((np.dot(X, weight)), y)
        weight = weight - learning_rate*(np.dot((X.T), grad))
        cost_history[i] = cost_function(X, y, weight)
    return weight, cost_history

weight = np.zeros((X.shape[1],1))
learning_rate = 0.0001
iterations = 5000
weight, cost_history = gradient_descent(X, y, weight, learning_rate, iterations)
print("\n weight: ", weight)
print("\n cost history: ", cost_history)