#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit

def get_data():
    load_data = datasets.load_iris()
    data = load_data.data[:,2:4]
    target = load_data.target
    return data, target

def data_deal(data, target):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    for train_index, test_index in split.split(data, target):
        train_set = data[train_index]
        test_set = data[test_index]
    return train_set, train_index, test_set, test_index

class Perceptron(object):
    def __init__(self, eta, n_iter):
        self.eta = eta            # Learning rate
        self.n_iter = n_iter      # Iter number
        self.weight = 0
        self.bias = 0

    def prediet(self, x, y):
        self.bias = y - self.weight * x

    def train(self, train, index, target):
        x_sum = y_sum = 0
        acc = 0                         # Accuracy of training
        j = 0
        for i in train:
            if np.any(target[index[j]] != 0):
                x_sum = x_sum + i[0]
                y_sum = y_sum + i[1]
            j += 1
        x_ave = x_sum / (len(train)*(2/3))      # Value center
        y_ave = y_sum / (len(train)*(2/3))
        for i in range(self.n_iter):
            match_target = []           # Store the judged target
            count = 0                   # Matching success times
            self.prediet(x_ave, y_ave)
            for y in train:
                if np.any(self.weight*y[0]+self.bias>=y[1]):
                    match_target.append(0)         # If ture, store the target(0)
                else:
                    match_target.append(2)
            for k in range(len(match_target)):
                if target[index[k]] == match_target[k]:
                    count += 1
            train_acc = count / (len(train)*(2/3))
            self._update_weights(acc, train_acc)
            acc = train_acc

        return acc

    def _update_weights(self, acc, train_acc):
        if acc <= train_acc:
            self.weight -= self.eta                # If ture, descending slope
        else:
            self.weight += self.eta                # If not, rising slope
        self.eta *= 0.9                            # In order to better fit the curve

def show(w, b, data):
    x = np.linspace(4, 6, 100)
    x1 = data[0:50,0]
    y1 = data[0:50,1]
    x2 = data[51:100,0]
    y2 = data[51:100,1]
    x3 = data[101:150,0]
    y3 = data[101:150,1]
    plt.plot(x, w*x+b, color='black')
    plt.scatter(x1, y1, c='r')
    plt.scatter(x2, y2)
    plt.scatter(x3, y3, c='y')
    plt.show()

def start_train(train_set, train_index, target):
    P = Perceptron(0.1, 100)
    acc = P.train(train_set, train_index, target)
    return P.weight, P.bias, acc

def test(w, b, test, index, target):
    match_target = []
    count = 0
    for y in test:
        if np.any(w * y[0] + b >= y[1]):
            match_target.append(0)
        else:
            match_target.append(2)
    for k in range(len(match_target)):
        if target[index[k]] == match_target[k]:
            count += 1
    acc = count / (len(test)*(2/3))
    return acc


if __name__ == '__main__':
    data, target = get_data()
    train_set, train_index, test_set, test_index = data_deal(data, target)

    w, b, acc = start_train(train_set, train_index, target)
    acc_t = test(w, b, test_set, test_index, target)
    print(f'weihgt = {w}, bias = {b}')
    print(f'Model accuracy: {acc*100}%')
    print(f'Test accuracy: {acc_t*100}%')
    show(w, b, data)