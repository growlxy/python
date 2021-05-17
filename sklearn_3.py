import numpy as np
import matplotlib.pyplot as plt

def generate_sample():
    X = np.linspace(0, 5, 100)
    y = np.power(X - 2.5, 2) - 1
    return X, y

class Sample:
    def __init__(self):
        self.w = 0
        self.w_list = np.array([0])

    def test(self, eta, n_iter):
        for i in range(n_iter):
            self.w = self.w - eta*2*(self.w-2.5)
            self.w_list = np.append(self.w_list, self.w)

    @property
    def values(self):
        return self.w_list

def show(X, y, s1, s2, s3, s4):
    plt.figure(figsize=(12, 12), dpi=100)

    plt.subplot(2, 2, 1)
    plt.title('eta = 0.01')
    plt.plot(X, y)
    plt.plot(s1, np.power(s1 - 2.5, 2) - 1, c='r')
    plt.scatter(s1, np.power(s1 - 2.5, 2) - 1, c='r')

    plt.subplot(2, 2, 2)
    plt.title('eta = 0.1')
    plt.plot(X, y)
    plt.plot(s2, np.power(s2 - 2.5, 2) - 1, c='r')
    plt.scatter(s2, np.power(s2 - 2.5, 2) - 1, c='r')

    plt.subplot(2, 2, 3)
    plt.title('eta = 0.8')
    plt.plot(X, y)
    plt.plot(s3, np.power(s3 - 2.5, 2) - 1, c='r')
    plt.scatter(s3, np.power(s3 - 2.5, 2) - 1, c='r')

    plt.subplot(2, 2, 4)
    plt.title('eta = 1')
    plt.plot(X, y)
    plt.plot(s4, np.power(s4 - 2.5, 2) - 1, c='r')
    plt.scatter(s4, np.power(s4 - 2.5, 2) - 1, c='r')

    plt.show()

if __name__ == '__main__':
    X, y = generate_sample()
    s1 = Sample()
    s1.test(0.01, 100)
    s2 = Sample()
    s2.test(0.1, 100)
    s3 = Sample()
    s3.test(0.8, 100)
    s4 = Sample()
    s4.test(1, 100)
    show(X, y, s1.values, s2.values, s3.values, s4.values)