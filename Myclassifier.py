# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:08:14 2018

@author: Shubham
"""


from scipy.spatial import distance

def euc(a,b): # It calculates the distance between two points for n dimensions
    return distance.euclidean(a,b)
"""
In My custom classifier i just need to define those methods 
which is defined in a general classifier like fit and predict
"""

class Myclassifier():
    def fit(self,X_train,Y_train):
        self.X_train = X_train     # memorize all traing data
        self.Y_train = Y_train
    def predict(self,X_train):
        prediction = []
        for row in X_train:
            label = self.classify(row)
            prediction.append(label)
        return prediction
            
    def classify(self,row):           #  this function finds the closest 
        closest_dist = euc(row,self.X_train[0]) # distance of test case
        closest_index = 0                       #  from already labelled data
        for i in range(1,len(self.X_train)):    #  and returns the label of
            temp_dist = euc(row,self.X_train[i]) # closest training example
            if temp_dist<closest_dist:
                closest_dist=temp_dist
                closest_index=i
        return self.Y_train[closest_index]
    


from sklearn import datasets
iris = datasets.load_iris()
"""
Like a Function we have X for input and Y for corresponding output
 f(X) = Y

"""

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

classifier = Myclassifier()
classifier.fit(X_train, Y_train)

"""
I just need to change above two lines for another new classifer

"""



predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print (accuracy_score(Y_test, predictions))

