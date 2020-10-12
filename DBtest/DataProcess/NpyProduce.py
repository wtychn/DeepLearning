import datetime

import matplotlib.pyplot as plt
import pandas as pd
import pyodbc


def csv_reading(fileName):
    # 读取csv文件
    rows = pd.read_csv(fileName)
    return rows


def is_continuous(t1, t2):
    # 判断是否换行
    datetime1 = datetime.datetime.strptime(t1[0:19], '%Y-%m-%d %H:%M:%S')
    datetime2 = datetime.datetime.strptime(t2[0:19], '%Y-%m-%d %H:%M:%S')
    return (datetime2 - datetime1).seconds > 3600


def time_difference(t1, t2):
    datetime1 = datetime.datetime.strptime(t1[0:19], '%Y-%m-%d %H:%M:%S')
    datetime2 = datetime.datetime.strptime(t2[0:19], '%Y-%m-%d %H:%M:%S')
    return (datetime2 - datetime1).seconds


if __name__ == '__main__':
    temperature = csv_reading("IronTempS.csv")
    velocity = csv_reading("IronVelocityS.csv")
    temIndex = 0
    velIndex = 0
    while temIndex < temperature.shape[0]:
        while abs(time_difference(temperature.iloc[temIndex][0], velocity.iloc[velIndex][0])) > 10:
            velIndex = velIndex + 1



