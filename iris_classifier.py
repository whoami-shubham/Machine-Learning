# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 02:32:51 2018

@author: Shubham
"""
from sklearn import datasets
iris = datasets.load_iris()
'''
 f(X) = Y
'''

X = iris.data
Y = iris.target

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =.5)

"""
split data into two parts for tarining and testing

"""
"""
from sklearn import tree
classifier = tree.DecisionTreeClassifier()

"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print (accuracy_score(Y_test, predictions))

