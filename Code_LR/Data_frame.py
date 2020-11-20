# from __future__ import division, print_function, unicode_literals
from sklearn import preprocessing
import numpy as np  
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error


# df = pd.read_csv('data/Admission_Predict.csv') 

# fig = sns.distplot(df['GRE Score'], kde=False)
# plt.title("Distribution of GRE Scores")
# plt.show()

# fig = sns.distplot(df['TOEFL Score'], kde=False)
# plt.title("Distribution of TOEFL Scores")
# plt.show()

# fig = sns.distplot(df['University Rating'], kde=False)
# plt.title("Distribution of University Rating")
# plt.show()

# fig = sns.distplot(df['SOP'], kde=False)
# plt.title("Distribution of SOP Ratings")
# plt.show()

# fig = sns.distplot(df['CGPA'], kde=False)
# plt.title("Distribution of CGPA")
# plt.show()

# plt.show()

# fig = sns.regplot(x="GRE Score", y="TOEFL Score", data=df)
# plt.title("GRE Score vs TOEFL Score")
# plt.show()

# sns.pairplot(df)
# plt.show()

# Đọc file data(csv)
data_frame = pd.read_csv('data/data_train.csv').values
data = data_frame[:, 1:9]
# print(data)
# chance_of_admit = data_frame[:, -1].reshape(-1, 1)
# print(Y_train)
# data_train = data_frame[:, 1:8]
# print(data_train) 

# Normalize
scaler = StandardScaler()
scaler.fit(data)
data_norm = scaler.fit_transform(data)
# print(data_norm)

chance_of_admit = data_norm[:, -1].reshape(-1, 1)
# print(chance_of_admit.shape)
data_train = data_norm[:, 0:7]
# print(data_train.shape)

# Đếm số các case
N = data_frame.shape[0]
# add_data = np.ones((N,1))
# Add 1
data1 = np.hstack((np.ones((N, 1)), data_train))
# print(data1.shape)



# code of me
w = np.random.randn(8)
# print(w.reshape(-1,1).shape)
number_of_iteration = 100000
learning_rate = 0.0001
cost = np.zeros((number_of_iteration,1))
w = np.expand_dims(w,0)
# Gradient Descent

for i in range(1, number_of_iteration):
    y_train = np.matmul(data1, w.reshape(-1,1))
    r = y_train - chance_of_admit
    cost[i] = (0.5/N)*np.sum(r*r)
    w -= learning_rate*1./N*np.matmul(r.T, data1)
    # w[0] -= learning_rate*(1./N)*np.sum(r)
    # w[1] -= learning_rate*(1./N)*np.sum(np.multiply(r, data1[:,1].reshape(-1,1)))
    # w[2] -= learning_rate*(1./N)*np.sum(np.multiply(r, data1[:,2].reshape(-1,1)))
    # w[3] -= learning_rate*(1./N)*np.sum(np.multiply(r, data1[:,3].reshape(-1,1)))
    # w[4] -= learning_rate*(1./N)*np.sum(np.multiply(r, data1[:,4].reshape(-1,1)))
    # w[5] -= learning_rate*(1./N)*np.sum(np.multiply(r, data1[:,5].reshape(-1,1)))
    # w[6] -= learning_rate*(1./N)*np.sum(np.multiply(r, data1[:,6].reshape(-1,1)))
    # w[7] -= learning_rate*(1./N)*np.sum(np.multiply(r, data1[:,7].reshape(-1,1)))
    print(cost[i])
print(w)


# Normal Equation
# A = np.dot(data1.T, data1)
# b = np.dot(dat
# data1.T, chance_of_admit)
# w1 = np.dot(np.linalg.pinv(A), b)
# print(w1)


# using Sklearn library
# regr = linear_model.LinearRegression(fit_intercept=False)
# regr.fit(data_train, chance_of_admit)
# # print(data_train.shape, chance_of_admit.shape, regr.coef_.shape)
# print('Solution found by scikit-learn: ',regr.coef_)
# # y_pred = np.matmul(data_, regr.coef_.T)
# print(regr.score(data_train, chance_of_admit), cost[number_of_iteration-1])

clf = linear_model.SGDRegressor(max_iter=10000, learning_rate='adaptive', eta0=0.0001)
# chance_of_admit = column_or_1d(chance_of_admit, warn=True)
clf.fit(data_train, chance_of_admit)
print('Solution found by scikit-learn: ',clf.coef_)
print(clf.score(data_train, chance_of_admit), cost[number_of_iteration-1])

# Test
df_test = pd.read_csv('data/data_test.csv').values
data_test = df_test[:, 1:9]
# print(data_test)
# Normalize
scaler = StandardScaler()
scaler.fit(data_test)
data_test_norm = scaler.fit_transform(data_test)
# y_test
y_test = data_test_norm[:, -1].reshape(-1, 1)
# print(y_test)
M = data_test.shape[0]
# print(M)
data__ = data_test_norm[:, 0:7]
# print(data__)
data = np.hstack((np.ones((M, 1)),data__))
y_pred = np.matmul(data, w.reshape(-1,1))
# print(y_pred)
r_test = y_pred - y_test
Loss = (0.5/M)*np.sum(r_test*r_test)

clf_test = linear_model.SGDRegressor(max_iter=10000, learning_rate='adaptive', eta0=0.0001)
clf_test.fit(data_test_norm, y_test)
print(clf_test.score(data_test_norm, y_test), Loss)