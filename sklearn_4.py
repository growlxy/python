import os
import struct
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def generate_sample():
    data = load_digits()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    y_train = (y_train == 6).astype(np.int)
    y_test = (y_test == 6).astype(np.int)
    return X_train, X_test, y_train, y_test


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    label = (labels == 6).astype(np.int)

    return images, label


def show(n, X_train, X_test, y_train, y_test):
    c_list = ['black', 'red', 'green', 'blue']

    for i in range(len(n)):
        model = LogisticRegression(max_iter=n[i])
        model.fit(X_train, y_train)
        y_score = model.decision_function(X_test)

        fpr, tpr, threshold = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)

        plt.figure(1, figsize=(12, 12), dpi=100)
        plt.subplot(2, 2, i + 1)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.plot(fpr, tpr, label=f'N = {n[i]} (AUC = {roc_auc:.4f})')
        plt.legend(loc='lower right')

        plt.figure(2, figsize=(8, 8), dpi=100)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(recall, precision, color=c_list[i], label=f'N = {n[i]}')
        plt.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    train_img, train_lab = load_mnist('mnist', kind='train')
    test_img, test_lab = load_mnist('mnist', kind='t10k')
    # X_train, X_test, y_train, y_test = generate_sample()
    n_list = [1, 2, 5, 10]

    # show(n_list, X_train, X_test, y_train, y_test)
    show(n_list, train_img, test_img, train_lab, test_lab)