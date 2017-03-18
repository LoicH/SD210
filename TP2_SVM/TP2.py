# -*- coding: utf-8 -*-
"""
@author: Loïc Herbelot
"""

from sklearn import datasets
from sklearn.svm import SVC
from random import sample

iris = datasets.load_iris()

"""This data sets consists of 3 different types of irises’ 
(Setosa, Versicolour, and Virginica) petal and sepal length, 
stored in a 150x4 numpy.ndarray
The rows being the samples and the columns being: 
Sepal Length, Sepal Width, Petal Length and Petal Width."""


X = iris.data 
y = iris.target

#We only want classes 1 & 2, and consider only the first 2 features:
X = X[y != 0, :2]
y = y[y != 0]

#indexes used for testing
i_test = sample(range(100), 50)
i_train = [i for i in range(100) if i not in i_test]

X_test = X[i_test]
X_train = X[i_train]

y_test = y[i_test]
y_train = y[i_train]

# fit the model with linear kernel
clf_lin = SVC(kernel='linear')
clf_lin.fit(X_train, y_train)

# predict labels for the test data base
y_pred = clf_lin.predict(X_test)

# check your score
score = clf_lin.score(X_test, y_test)
print('Score with linear kernel: %s' % score)

# fit the model with poly kernel
clf_poly = SVC(kernel='poly')
clf_poly.fit(X_train, y_train)

# predict labels for the test data base
y_pred = clf_poly.predict(X_test)

# check your score
score = clf_poly.score(X_test, y_test)
print('Score with polynomial kernel: %s' % score)



