import pandas as pd
import numpy as np
from pandas import DataFrame

data = pd.read_csv('Admission_Predict.csv')
length = len(data)
print(length)
data_train = data[:int(length*0.8)]
# print(data_train)
data_test = data[int(length*0.8):length]
data_train.to_csv('data/data_train.csv', index=False, header=True)
data_test.to_csv('data/data_test.csv', index=False, header=True)