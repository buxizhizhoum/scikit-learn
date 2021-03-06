#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
refer: https://blog.csdn.net/pipisorry/article/details/52128222?utm_source=blogkpcl6
"""
import numpy as np
from sklearn import neighbors
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV


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
    knn = neighbors.KNeighborsClassifier(n_neighbors=7)

    # optimize with grid search
    params = {"n_neighbors": np.arange(1, 7),
              "metric": ["euclidean", "cityblock"]}
    grid = GridSearchCV(estimator=knn, param_grid=params)
    grid.fit(X_train, y_train)
    print(grid.best_score_)
    print(grid.best_estimator_)
