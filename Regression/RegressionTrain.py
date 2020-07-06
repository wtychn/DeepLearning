import math
import sys
import pandas as pd
import numpy as np

data = pd.read_csv('../dataSources/train.csv', encoding='big5')

# 取所有行三列往后的值，去除表头说明
data = data.iloc[:, 3:]
# 将NR置为0
data[data == 'NR'] = 0
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    # 每个月取20天的数据
    for day in range(20):
        # 把原始数据中的4320*18行的数据转化为18*12的数据
        sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    month_data[month] = sample

# 前九个小时作为预测特征，每九小时作为一行（一个data）
x = np.empty([12 * 471, 18 * 9], dtype=float)
# 第十个小时作为预测结果，目的是拟合出y=f(x)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(
                -1)  # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value

# 标准化
mean_x = np.mean(x, axis=0)  # 18 * 9 均值
np.save("mean_x", mean_x)
std_x = np.std(x, axis=0)  # 18 * 9 标准差
np.save("std_x", std_x)
for i in range(len(x)):  # 12 * 471
    for j in range(len(x[0])):  # 18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]

dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)
learning_rate = 2
iter_time = 10000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)  # RMSE均方根误差公式
    if (t % 1000 == 0):
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)
