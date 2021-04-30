import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit

def get_data():
    load_data = datasets.load_iris()
    data = load_data.data
    target = load_data.target
    return data, target

def get_deal(data, target):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    for train_index, test_index in split.split(data, target):
        train_set = data[train_index]
        test_set = data[test_index]
    return train_set, test_set

if __name__ == '__main__':
    data, target = get_data()
    train_set, test_set = get_deal(data, target)

# https://blog.csdn.net/qq_43923588/article/details/107672879
# https://blog.csdn.net/weixin_41987016/article/details/107626615