# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 08:28:33 2018

@author: Shubham

"""
"""
     Training data
-------------------------
|Weight | Texture | Label
| 150g  | Bumpy   | Orange
| 170g  | Bumpy   | Orange
| 140g  | Smooth  | Apple
| 130g  | Smooth  | Apple
|
"""



from sklearn import tree
features = [[140,1],[130,1],[150,0],[170,0]]  
labels = ["apple","apple","orange","orange"]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
print(clf.predict([[160,0]]))
