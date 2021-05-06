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
        self.eta = eta                  # Learning rate
        self.n_iter = n_iter            # Iter number
        self.weight = 0
        self.bias = 0

    def prediet(self, y):
        return self.weight * y + self.bias

    def train(self, train, index, target, flag):
        x_sum = y_sum = 0
        acc = 0                         # Accuracy of training
        j = 0
        for i in train:
            if np.any(target[index[j]] != flag):
                x_sum = x_sum + i[0]
                y_sum = y_sum + i[1]
            j += 1
        x_ave = x_sum / (len(train)*(2/3))         # Value center
        y_ave = y_sum / (len(train)*(2/3))
        for i in range(self.n_iter):
            match_target = []           # Store the judged target
            count = 0                   # Matching success times
            self.bias = y_ave - self.weight * x_ave
            for y in train:
                if np.any(self.prediet(y[0])>=y[1]):
                    match_target.append(0)         # If ture, store the target(0)
                else:
                    match_target.append(2-0.5*flag)
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
    x = np.linspace(4, 7, 100)
    X = np.linspace(1, 3.5, 100)
    x1 = data[0:50,0]
    y1 = data[0:50,1]
    x2 = data[51:100,0]
    y2 = data[51:100,1]
    x3 = data[101:150,0]
    y3 = data[101:150,1]

    plt.plot(x, w[0]*x+b[0], color='black', label='line 1')
    plt.plot(X, w[1]*X+b[1], color='black', label='line 2')
    plt.scatter(x1, y1, c='r')
    plt.scatter(x2, y2)
    plt.scatter(x3, y3, c='y')
    plt.legend(loc='upper right')
    plt.show()

def start_train(train_set, train_index, target):
    P1 = Perceptron(0.1, 100)
    acc_1 = P1.train(train_set, train_index, target, 0)
    P2 = Perceptron(0.1, 100)
    acc_2 = P2.train(train_set, train_index, target, 2)

    weight = [P1.weight, P2.weight]
    bias = [P1.bias, P2.bias]
    acc = [acc_1, acc_2]
    return weight, bias, acc

def test(w, b, test, index, target):
    acc_t = []
    for i in range(0, 2):
        match_target = []
        count = 0
        for y in test:
            if np.any(w[i] * y[0] + b[i] >= y[1]):
                match_target.append(0)
            else:
                match_target.append(2-i)
        for k in range(len(match_target)):
            if target[index[k]] == match_target[k]:
                count += 1
        acc = count / (len(test)*(2/3))
        acc_t.append(acc)

    return acc_t

def cheak(w, b, x, y):
    target = ['setosa', 'versicolor', 'virginica']
    line_1 = w[0] * x + b[0]
    line_2 = w[1] * x + b[1]
    if y > line_1:
        print(target[2])
    elif y > line_2:
        print(target[1])
    else:
        print(target[0])

if __name__ == '__main__':
    data, target = get_data()
    train_set, train_index, test_set, test_index = data_deal(data, target)

    w, b, acc = start_train(train_set, train_index, target)
    acc_t = test(w, b, test_set, test_index, target)
    print(f'line 1:\n'
          f'weihgt = {w[0]}, bias = {b[0]}\n'
          f'Model accuracy: {acc[0]*100}%\n'
          f'Test accuracy: {acc_t[0]*100}%')
    print(f'line 2:\n'
          f'weihgt = {w[1]}, bias = {b[1]}\n'
          f'Model accuracy: {acc[1]*100}%\n'
          f'Test accuracy: {acc_t[1]*100}%')

    # show(w, b, data)

    print('-------------------------------------------------')

    x, y = map(float, input('Please input:').split())
    cheak(w, b, x, y)
    