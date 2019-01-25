# -*- coding: utf-8 -*-
"""

@author: FaizMohammad
"""

#importing all the required Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier


#insert iris_data from sklearn.datasets and separate features and target
#let's take X = features and y =  Target 

iris = load_iris()
X = iris.data
y= iris.target

#Now let's divide the data for traning and testing with test-size= 20%
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)

#Select Decision tree Classifier algorithm for model fitting
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)

#Now check the Model prediction with testing data
predict = classifier.predict(X_test)

#and at last check the Model Accuracy 
cm= accuracy_score(pred,y_test)

#plot (X VS y )graph where X-label is sepal length(cm) and y-label is sepal width(cm)
feature1=0
feature2=1
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.figure(figsize=(5,4))
plt.scatter(X[:,feature1],X[:,feature2],c=y)
plt.colorbar(ticks=[0,1,2],format=formatter)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()