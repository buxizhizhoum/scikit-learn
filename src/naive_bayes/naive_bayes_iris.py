#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
refer: https://blog.csdn.net/pipisorry/article/details/52128222?utm_source=blogkpcl6
"""
from sklearn import naive_bayes
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


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

    model = naive_bayes.GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
