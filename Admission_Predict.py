import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
dataFrame = pd.read_csv('Admission_Predict.csv')
dataFrame_to_numpy = dataFrame.to_numpy()

# split dataset to train data and test data (3 / 1)
data = dataFrame_to_numpy[0:300,:]
test = dataFrame_to_numpy[300:400,:]

# calculate w from train data
one = np.ones((len(data), 1))
y = data[:,8]
Xbar = np.concatenate((one, data[:,1:8]), axis = 1)
XbarT_dot_Xbar = np.dot(Xbar.T, Xbar)
XbarT_dot_y = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(XbarT_dot_Xbar), XbarT_dot_y)

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

# Compare two results
print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by (Tuan Vu)     : ', w.T)

# calculate total error with test data
error = 0
for i in range(100):
    x = np.insert(test[i, 1:8], 0, 1)
    error = error + 0.5 * (test[i, 8] - np.dot(x, w))**2
    
print('total error for test dataset = ', error)