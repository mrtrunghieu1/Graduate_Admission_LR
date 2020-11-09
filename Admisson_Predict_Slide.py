import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
dataFrame = pd.read_csv('Admission_Predict.csv')
learning_rate = 1
dataFrame_to_numpy = dataFrame.to_numpy()
data = dataFrame_to_numpy[0:300,:]
test = dataFrame_to_numpy[300:400,:]
one = np.ones((len(data), 1))
y = data[:,8]
Xbar = np.concatenate((one, data[:,1:8]), axis = 1)
#w = [1, 1, 1, 1, 1, 1, 1, 1] #initial w
w = [-1.27488267,  0.00176755,  0.00290322,  0.00771492, -0.00531496,  0.02828439, 0.11736896,  0.01922889]
for i in range(300):
    E = 0.5*((np.dot(Xbar[i], w) - y[i])**2)
    w = w + (y[i] - np.dot(Xbar[i], w))*Xbar[i]
