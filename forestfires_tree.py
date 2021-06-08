import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('forestfires.csv')
data = df[['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind']]
target = (df['area'] > 0).astype(np.int)
columns = data.columns.values

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)

def determine_continuous(data, columns):
    con_list = []
    for i in columns:
        if isinstance(data[i], str):
            con_list.append(True)
        else:
            con_list.append(None)

    return con_list

def entrop(data, target):
    feature_list = list(set(data))
    ent = 0
    for feature in feature_list:
        True_num, False_num = 0, 0
        for sample in zip(data, target):
            if sample[0] == feature:
                if sample[1] == 1:
                    True_num += 1
                else:
                    False_num += 1

        True_pro = True_num / (True_num + False_num)
        False_pro = False_num / (True_num + False_num)
        if True_pro != 0 and False_pro != 0:
            ent_ = -(True_pro * np.log2(True_pro) + False_pro * np.log2(False_pro))
        else:
            ent_ = 0
        ent += ent_ * ((True_num + False_num) / len(data))

    return ent

def entrop_i(data):
    feature_list = list(set(data))
    feature_dict = dict(zip(feature_list, np.zeros(len(feature_list))))
    for feature in data:
        for key in feature_dict.keys():
            if key == feature:
                feature_dict[key] += 1

    ent_i = 0
    for value in feature_dict.values():
        value_pro = value / len(data)
        ent_i += value_pro * np.log2(value_pro)

    return -ent_i

def gain(ent, ent_, ent_i):
    gain_ = ent - ent_
    gain_ratio = gain_ / ent_i
    return gain_, gain_ratio

# con_list = determine_continuous(X_train.loc[0], columns)
ent = entrop_i(y_train)
ent_ = entrop(X_train['month'], y_train)
gain_, gain_ratio = gain(ent, ent_, entrop_i(X_train['month']))
print(gain_, gain_ratio)
