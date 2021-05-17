import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def show(w, b, x_scale):
    plt.axis([-4.10, 2.76, -4.48, 4.85])
    for i in range(len(y)):
        if y[i] == 0:
            plt.scatter(x_scale[i, 0], x_scale[i, 1], c='purple')
        else:
            plt.scatter(x_scale[i, 0], x_scale[i, 1], c='yellow')
    x = np.linspace(-4, 4, 200)
    line = (-b - w[:, 0] * x) / w[:, 1]
    line_lower = (-1 - b - w[:, 0] * x) / w[:, 1]
    line_upper = (1 - b - w[:, 0] * x) / w[:, 1]
    plt.plot(x, line, 'k-')
    plt.plot(x, line_lower, 'k--')
    plt.plot(x, line_upper, 'k--')

    plt.show()

df = pd.read_csv('diabetes.csv')
X = df[['Glucose', 'BMI']]
y = df[['Outcome']].values.ravel()
x_scale = preprocessing.scale(X)

cls = SVC(kernel='linear')
cls.fit(x_scale, y)
w = cls.coef_
b = cls.intercept_

# show(w, b, x_scale)

score = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.2, random_state=i)
    score.append(cls.score(X_test, y_test))
print(f'Average accuracy of 10 prediction tests:{sum(score)/len(score)}')
print(f'All the accuracy of 10 prediction tests:\n{score}')