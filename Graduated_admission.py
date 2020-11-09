import pandas as pd 
import numpy as np
import csv
import re
from matplotlib import pyplot as plt
from sklearn import datasets,linear_model

data=np.zeros((400,8))
label=np.zeros((400,1),dtype='float64')
# w=np.random.rand(1,7)*math.pow(10,2)
w=np.random.uniform(-50,500,size=(1,7))
bias=np.random.rand(280,1)
bias_test=np.random.rand(120,1)
eta=0.025
eta_list=np.linspace(1e-06,0.2,100)
loss_list=[]
# Read data
i=0
with open('Admission_Predict.csv','r') as file:
    data_source=csv.reader(file)
    for row in data_source:
        if bool(re.search(r'\s',row[0])):
            continue
        data[i,:]=row[0:8]
        label[i,:]=row[8]
        i+=1

norm_data=data/np.linalg.norm(data)
norm_label=label/np.linalg.norm(label)

X_train,X_test=norm_data[0:280,1:8],norm_data[280:,1:8]
Y_train,Y_test=label[0:280,:],label[280:,:]

# simuluate between loss and eta relationship
# for i in eta_list:
#     y_pre=np.dot(X_train,w.T)+bias
#     grad=np.dot(X_train.T,(y_pre-Y_train))
#     loss=0.5*np.sum(np.power(y_pre-Y_train,2))
#     w=w-i*grad.T
#     loss_list.append(loss)


for i in range(500):
    y_pre=np.dot(X_train,w.T)+bias              # caculate output y
    grad=np.dot(X_train.T,(y_pre-Y_train))      # caculate gradient descent
    loss=0.5*np.sum(np.power(y_pre-Y_train,2))  # loss function
    w=w-eta*grad.T
    loss_list.append(loss)                      # update weight
    if i%50==0:
        print(" Iteration %2.f is %2.f" %(i,loss))
y_test_pre=np.dot(X_test,w.T) +bias_test
loss_test=0.5*np.sum(np.power(y_test_pre-Y_test,2))

# using Sklearn library
regr=linear_model.LinearRegression(fit_intercept=False)
regr.fit(X_train,Y_train)
print('Solution found by scikit-learn: ',regr.coef_)
print('Solution found by me:',w)
print('Loss of test data: ',loss_test)
# plt.plot(eta_list,loss_list)
plt.plot(loss_list)
plt.show()



