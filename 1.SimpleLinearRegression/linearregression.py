# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:55:49 2018

@author: Faizmohammad
"""
#importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the datsets
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#import the Linear Model Algoritm from sklearn and fit the data in it
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size= 1/3, random_state=0)


#fitting into regression model
from sklearn.linear_model import LinearRegression
regresseor= LinearRegression()
regresseor.fit(X_train,y_train)

#predict the results from the test set
ytest_pred= regresseor.predict(X_test)
ytrain_pred = regresseor.predict(X_train)

#visulizing traning set
plt.scatter(X_train, y_train, color= "red")
plt.plot(X_train, ytrain_pred, color= "Blue")
plt.title("Salary VS Expereinces (Traning data) ")
plt.xlabel("years of Experiences")
plt.ylabel("Salary")
plt.show()

#visualising the test set
plt.scatter(X_test,y_test, color= "Red")
plt.plot(X_train,ytrain_pred, color= "Blue")
plt.title("Experiences vs Salary (Test data)")
plt.xlabel("Years of Experiences")
plt.ylabel("Salary")
plt.show()
