import datetime

import sys
import matplotlib.pyplot as plt
import pandas as pd
import pyodbc
import numpy as np
from tqdm import tqdm


def is_continuous(t1, t2):
    # 判断是否为一个循环
    return time_difference(t1, t2) < 3600


def time_difference(t1, t2):
    dt1 = datetime.datetime.strptime(t1, '%Y-%m-%d %H:%M:%S')
    dt2 = datetime.datetime.strptime(t2, '%Y-%m-%d %H:%M:%S')
    return dt2.timestamp() - dt1.timestamp()


def time_classification(t):
    if t < 100:
        return 0
    elif t < 110:
        return 1
    elif t < 120:
        return 2
    elif t < 130:
        return 3
    elif t < 140:
        return 4
    elif t < 150:
        return 5
    elif t < 160:
        return 6
    elif t < 170:
        return 7
    elif t < 180:
        return 8
    elif t < 190:
        return 9
    elif t < 1000:
        return 10


if __name__ == '__main__':
    # 读取数据并归一化
    temperature = pd.read_csv("IronTempS.csv", usecols=['time', 'temp'])
    temperature['temp'] = (temperature['temp'] - temperature['temp'].min()) / (
                temperature['temp'].max() - temperature['temp'].min())
    temperature = temperature[(temperature['temp'] < 0.9) & (temperature['temp'] > 0.2)]

    velocity = pd.read_csv("IronVelocityS.csv", usecols=['time', 'vel'])
    velocity = velocity[velocity['vel'] > 0]
    velocity['vel'] = (velocity['vel'] - velocity['vel'].min()) / (
                velocity['vel'].max() - velocity['vel'].min())

    temIndex = 0
    velIndex = 0

    data_x = []  # n * x1
    x1 = np.zeros((100, 2))  # 100 * [tem, vel]
    data_y = []
    data_classification = []
    hun_count = 0
    data_count = 0  # 数据总量

    first_time = ""
    min_time = sys.maxsize
    max_time = 0

    # total参数设置进度条的总长度
    pbar = tqdm(total=temperature.shape[0])
    while temIndex < temperature.shape[0]:
        if is_continuous(temperature.iloc[temIndex][0], temperature.iloc[temIndex + 1][0]):
            if hun_count < 100:
                # vel时间与tem相匹配
                line_count = velIndex - 3
                min_diff = sys.maxsize
                while line_count < 0:
                    line_count += 1
                while line_count < velocity.shape[0] and line_count < velIndex + 4:
                    diff = abs(time_difference(velocity.iloc[line_count][0], temperature.iloc[temIndex][0]))
                    if(diff < min_diff):
                        min_diff = diff
                        velIndex = line_count
                    line_count += 1

                x1[hun_count, 0] = temperature.iloc[temIndex][1]
                x1[hun_count, 1] = velocity.iloc[velIndex][1]
                velIndex += 1
                # 记录一个表的初始时间
                if hun_count == 0:
                    first_time = temperature.iloc[temIndex][0]
            hun_count += 1
        else:
            if hun_count >= 100:
                # 到了时间断点且数据大于100个就统计时间并将数据填入数据集
                data_x.append(x1)
                time = time_difference(first_time, temperature.iloc[temIndex - 1][0]) / 60
                data_y.append(time)
                data_classification.append(time_classification(time))
                data_count += 1
            x1 = np.zeros((100, 2))
            hun_count = 0

        temIndex += 1
        # 每次更新进度条的长度
        pbar.update(1)

    print('共有', data_count, '组数据')

    np.save('data_x.npy', data_x)
    np.save('data_y.npy', data_y)
    np.save('data_c.npy', data_classification)
