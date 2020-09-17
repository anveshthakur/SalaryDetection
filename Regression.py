# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:55:31 2020

@author: dell-pc
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[: , :-1].values  
Y = dataset.iloc[: , 1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3, random_state = 0)

#simple Linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


#Predicting the values
y_pred = regressor.predict(X_test) 

#visualising the training set
plt.scatter(X_train,Y_train,color = "Red")
plt.plot(X_train, regressor.predict(X_train))
plt.title('ExperiencevsSalary')
plt.xlabel('experience')
plt.ylabel('salary')

#visualising the test set
plt.scatter(X_test, Y_test,color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = "blue")
plt.title('salary vs exp')
plt.xlabel('experience')
plt.ylabel('salary')