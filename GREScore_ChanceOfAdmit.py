import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataFrame = pd.read_csv('Admission_Predict.csv')
data = dataFrame.to_numpy()
GRE_Score = data[:,1]
TOEFL_Score = data[:,2]
University_Rating = data[:,3]
SOP = data[:,4]
LOR = data[:,5]
CGPA = data[:,6]
Research = data[:,7]
Chance_of_Admit = data[:,8]
plt.plot(GRE_Score.T, Chance_of_Admit.T, 'ro')
plt.xlabel('GRE Score')
plt.ylabel('Chance of Amit')