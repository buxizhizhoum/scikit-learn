#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
refer: https://blog.csdn.net/pipisorry/article/details/52128222?utm_source=blogkpcl6
"""
from sklearn import tree
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    print(X.shape, y.shape)
    # Split arrays or matrices into random train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Standardize features by removing the mean and scaling to unit variance
    scaler = preprocessing.StandardScaler().fit(X_train)
    # Perform standardization by centering and scaling
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # decision tree
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    print(accuracy_score(y_test, y_pred))
