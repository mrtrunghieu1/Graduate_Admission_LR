# from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import utils
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split, GridSearchCV # for splitting data
from sklearn.metrics import accuracy_score, mean_squared_error # for evaluating results
from math import sqrt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from my_weight import myweight
# from sklearn.utils import shuffle


path = 'D:\\Projects\\Python projects\\Machine Learning Lab\\Son_Code_KNN\\movie.csv'
movie_df = pd.read_csv(path)
movie_df = movie_df.dropna()
movie_df = movie_df.reset_index(drop=True)

train_data = movie_df.loc[:799, :]
test_data = movie_df.loc[800:, :]

X_train = train_data.loc[:][['budget', 'revenue', 'runtime', 'popularity', 'vote_counts']]
y_train = train_data.loc[:, ['vote_averages']]
X_test = test_data.loc[:][['budget', 'revenue', 'runtime', 'popularity', 'vote_counts']]
y_test = test_data.loc[:, ['vote_averages']]

scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
y_train_normalized = scaler.fit_transform(y_train)
X_test_normalized = scaler.fit_transform(X_test)
y_test_normalized = scaler.fit_transform(y_test)


params = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12]}
knn = neighbors.KNeighborsRegressor()

model_grid = GridSearchCV(knn, params, cv=5)
model_grid.fit(X_train_normalized,y_train_normalized)
k_best = model_grid.best_params_['n_neighbors']

# prediction
knn1 = neighbors.KNeighborsRegressor(n_neighbors = k_best, p = 2)
knn1.fit(X_train_normalized, y_train_normalized)
y_pred = knn1.predict(X_test_normalized)
print(y_pred * 10)


rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train_normalized, y_train_normalized)  #fit the model
    pred = model.predict(X_test_normalized) #make prediction on test set
    error = sqrt(mean_squared_error(y_test_normalized,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)

#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()
plt.show()
