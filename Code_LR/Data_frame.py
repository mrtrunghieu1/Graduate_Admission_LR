from __future__ import division, print_function, unicode_literals
from sklearn import preprocessing
import numpy as np  
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn import datasets,linear_model 

# Đọc file data(csv)
data_frame = pd.read_csv('data/data_train.csv').values
data = data_frame[:, 1:9]
# print(data)
scaler = StandardScaler()

scaler.fit(data)
data_norm = scaler.fit_transform(data)
# print(data_norm)
chance_of_admit = data_norm[:, -1].reshape(-1, 1)
# print(chance_of_admit)
data_train = data_norm[:, 0:7]

# print(data_train)
# Đếm số các case
N = data_frame.shape[0]
# add_data = np.ones((N,1))
data1 = np.hstack((np.ones((N, 1)), data_train))
# print(data1)
# print(data1[0])

# print(chance_of_admit)
# # Chọn cột và reshape lại 
# gre_score = data_frame[:, 1].reshape(-1, 1)
# # mean = np.mean(gre_score)
# # std = np.std(gre_score)
# # gre_score_nor = (gre_score-mean)/std
# gre = gre_score/np.linalg.norm(gre_score)
# gre_score_nor = preprocessing.scale(gre_score)
# print(gre_score_nor)
# tofel_score = data_frame[:, 2].reshape(-1, 1)
# tofel_score = preprocessing.scale(tofel_score)
# university_rating = data_frame[:, 3].reshape(-1, 1)
# university_rating = preprocessing.scale(university_rating)
# sop = data_frame[:, 4].reshape(-1, 1)
# sop = preprocessing.scale(sop)
# lor = data_frame[:, -4].reshape(-1, 1)
# lor = preprocessing.scale(lor)
# cgpa = data_frame[:, -3].reshape(-1, 1)
# cgpa = preprocessing.scale(cgpa)
# research = data_frame[:, -2].reshape(-1, 1)
# research = preprocessing.scale(research)
# chance_of_admit = data_frame[:, -1].reshape(-1, 1)
# chance_of_admit = preprocessing.scale(chance_of_admit)


# code of me


w = np.random.randn(8)
number_of_iteration = 10000
learning_rate = 0.0001
cost = np.zeros((number_of_iteration,1))

# print(w.T)
# print(w.reshape(-1,1))
# y_train = np.dot(data1, w.reshape(-1,1))
# r = chance_of_admit - y_train
# print(y_train)
# A = (1./N)*np.sum(np.multiply(r, data1[:,1].reshape(-1,1)))
# print(A)
# print(data1, data1[:,1].reshape(-1,1))

for i in range(1, number_of_iteration):
    y_train = np.dot(data1, w.reshape(-1,1))
    r = y_train - chance_of_admit
    cost[i] = (0.5/N)*np.sum(r*r)
    w[0] -= learning_rate*(1./N)*np.sum(r)
    w[1] -= learning_rate*(1./N)*np.sum(np.multiply(r, data1[:,1].reshape(-1,1)))
    w[2] -= learning_rate*(1./N)*np.sum(np.multiply(r, data1[:,2].reshape(-1,1)))
    w[3] -= learning_rate*(1./N)*np.sum(np.multiply(r, data1[:,3].reshape(-1,1)))
    w[4] -= learning_rate*(1./N)*np.sum(np.multiply(r, data1[:,4].reshape(-1,1)))
    w[5] -= learning_rate*(1./N)*np.sum(np.multiply(r, data1[:,5].reshape(-1,1)))
    w[6] -= learning_rate*(1./N)*np.sum(np.multiply(r, data1[:,6].reshape(-1,1)))
    w[7] -= learning_rate*(1./N)*np.sum(np.multiply(r, data1[:,7].reshape(-1,1)))
    print(cost[i])
print(w)
# Normal Equation
# A = np.dot(data1.T, data1)
# b = np.dot(data1.T, chance_of_admit)
# w1 = np.dot(np.linalg.pinv(A), b)
# print(w1)


# using Sklearn library
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(data1,chance_of_admit)
print('Solution found by scikit-learn: ',regr.coef_) 
