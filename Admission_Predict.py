import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets, linear_model
from pandas.plotting import scatter_matrix

# read data and plot
df = pd.read_csv("Admission_Predict.csv")
pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(12,8))

# convert to numpy array
df = df.values

# split train set and test set
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)
df = train_set.copy()

# calculate w
one = np.ones((len(df),1))
Xbar = df[:, 1:8]
Xbar = np.concatenate((one, Xbar), axis=1)
y = df[:,8].reshape(len(df),1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)

# loss 
e_train = 0
for i in range(len(df)):
    e_train += (np.subtract(y[i], np.dot(Xbar[i], w)))**2
error_train = e/2

# test
one = np.ones((len(test_set),1))
X_test = test_set[:, 1:8]
X_test = np.concatenate((one, X_test), axis=1)
y_test = test_set[:,8].reshape(len(test_set),1)

A_test = np.dot(X_test.T, X_test)
b_test = np.dot(X_test.T, y_test)
a
e_test = 0
for i in range(len(test_set)):
    e_test += (np.subtract(y_test[i], np.dot(X_test[i], w)))**2
error_test = e_test/2


print("w là", w)
print("error_train:", error_train)
print("error_test:", error_test)