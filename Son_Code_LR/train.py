import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cost_function import cost_function
from gradient_descent import gradient_descent
from features_normalize import features_normalize


data = pd.read_csv('D:\\Projects\\Python projects\\Machine Learning Lab\\Graduate_Admission_LR\\data\\data_train.csv')
data = data.values
X = data[:, 1:(data.shape[1] - 1)]
y = data[:, -1]

print('First 10 examples from the dataset:')
print(X[:10, :])
print(y[:10])

print('Normalizing Features ...')
X, mu, sigma = features_normalize(X)
print('X after normalization: ')
print(X)

# Add biases to X
X = np.hstack((np.ones((X.shape[0], 1)), X))

print('Running gradient descent ...')
alpha = 0.0001 # change this
num_iters = 5000 # change this

W = np.zeros((X.shape[1], 1))
W, J_history = gradient_descent(X, y, W, alpha, num_iters)

print('Parameters: \n', W)
print(J_history)
plt.plot(np.array(range(1, num_iters + 1)), J_history, linewidth=2)
plt.show()

