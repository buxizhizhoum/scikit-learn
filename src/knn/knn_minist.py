#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
knn to classify MINIST
refer: https://mp.weixin.qq.com/s?src=11&timestamp=1539739507&ver=1187&signature=UCqG*-56reh3e43YoDD3BrshK8anSbT8lVjxj8tera3q43LUwNCqlZ1MnEj*-SlyH41*vLoBwxZ3GMonV9Ylkg73Sf3HpG5SfG-SO898USiThLWMhufBaM6tkgcTAVFS&new=1
"""
import numpy as np

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def mk_dataset(size):
    train_img = [data[i] for i in index[:size]]
    train_x = np.array(train_img)
    target_label = [target[i] for i in index[:size]]
    target_y = np.array(target_label)
    return train_x, target_y


if __name__ == "__main__":
    minist = datasets.fetch_mldata("MNIST original")
    data, target = minist.data, minist.target
    print(data.shape, target.shape)

    index = np.random.choice(len(target), 70000, replace=False)

    fit_x, fit_y = mk_dataset(50000)

    test_img = [data[i] for i in index[60000:70000]]
    test_x = np.array(test_img)
    test_label = [target[i] for i in index[60000:70000]]
    test_y = np.array(test_label)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(fit_x, fit_y)

    test_pred = classifier.predict(test_x)
    print(test_pred)
    print(accuracy_score(test_y, test_pred))
    print(classification_report(test_y, test_pred))




