import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
dataFrame = pd.read_csv('Admission_Predict.csv').to_numpy()
# chia dữ diệu trước quá trình train và quá tình test 
lengh = int(len(dataFrame)*0.6)

# data trian
data_train = dataFrame[0:lengh,:8]
chain_of_admit_train = dataFrame[:lengh,-1] 
for i in range(lengh):   #tạo matrix_train có cột wo = 1
    data_train[i][0] = 1
Xbar_train = data_train
# print(Xbar_train)
# print(data_train)

# data test
data_test = dataFrame[lengh:,:8]
chain_of_admit_test = dataFrame[lengh:,-1]
for i in range(len(data_test)):
    data_test[i][0] = 1
Xbar_test = data_test
# print(chain_of_admit_test)
# print(Xbar_test)

# Train
def Train():
    Xbar_T_dot_Xbar = np.dot(Xbar_train.T, Xbar_train)
    Xbar_T_dot_result = np.dot(Xbar_train.T, chain_of_admit_train)
    w = np.dot(np.linalg.pinv(Xbar_T_dot_Xbar), Xbar_T_dot_result)
    return w

#using tool
def Train_tool():
    w_tool = linear_model.LinearRegression(fit_intercept=False) 
    w_tool.fit(Xbar_train, chain_of_admit_train)
    return w_tool


# loss function
def loss_func(): 
    matrix_result = np.dot(Xbar_test , Train())
    matrix_loss_E = chain_of_admit_test - matrix_result
    for i in range(len(matrix_loss_E)):
        matrix_loss_E[i] = pow(matrix_loss_E[i],2)
    E = (0.5/len(data_test))*sum(matrix_loss_E)
    return E

def main():
    print("result by Phong: w = ", Train())
    print("result by Tool: w = ", Train_tool().coef_)
    print("Error_data_test: = ", loss_func())


if __name__ == '__main__':
    main()
    
            
