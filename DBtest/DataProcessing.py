import io
import datetime
import numpy as np
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt
import pyodbc

sqlStr = "select top 1000 t1.时间, 铁水红外温度, 铁水流速, 瞬时出铁量 \
          from (select 时间, 铁水红外温度 from dbo.IronTemp where 铁口号 = 1 and 时间 > '2020-08-01') as t1 \
          left join (select 时间, 铁水流速, 瞬时出铁量 from dbo.IronVelocity where 铁口号 = 1 and 时间 > '2020-08-01') as t2 \
          on datediff(second ,t1.时间,t2.时间) < 4 and datediff(second ,t1.时间,t2.时间) > -4"


def sql_connection(sql):
    # 读取数据库
    driver = 'SQL Server Native Client 11.0'  # 因版本不同而异
    server = 'csu-tech.mynatapp.cc,65141'
    user = 'sa'
    password = '0000'
    database = 'IronWData'

    conn = pyodbc.connect(driver=driver, server=server, user=user, password=password, database=database)

    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()  # list
    conn.close()

    return rows


def csv_reading(fileName):
    # 读取csv文件
    rows = pd.read_csv(fileName)
    return rows


def is_continuous(t1, t2):
    # 判断是否换行
    datetime1 = datetime.datetime.strptime(t1[0:19], '%Y-%m-%d %H:%M:%S')
    datetime2 = datetime.datetime.strptime(t2[0:19], '%Y-%m-%d %H:%M:%S')
    return (datetime2 - datetime1).seconds > 3600


if __name__ == '__main__':
    data = csv_reading("testData.csv")
    temp = []
    velocity = []
    weight = []
    x = []
    index = 0
    group_count = 1
    for i in range(data.shape[0]):
        if i != 0 and is_continuous(data.iloc[i - 1, 0], data.iloc[i, 0]):
            print("这是第", group_count, "组数据，共有", index, "个元素")
            plt.xlabel('Time', color='blue')  # 坐标轴标题
            plt.ylabel('Value', color='blue')
            plt.plot(x, temp, label='temp', color='r', linewidth=3.0)  # 输入温度数据
            plt.plot(x, velocity, label='velocity', color='b', linewidth=2.0)  # 输入流速数据
            plt.plot(x, weight, label='weight', color='y', linewidth=2.0)  # 输入重量数据
            plt.grid(alpha=0.3, linestyle=':')  # 画网格
            plt.legend(loc='best')  # 画图例
            plt.show()
            temp = []
            velocity = []
            weight = []
            x = []
            index = 0
            group_count = group_count + 1

        # 温度信息减小数值，将低位数值作为增益保留原来的波动程度
        temp.append(data.iloc[i, 1] / 400 + data.iloc[i, 1] % 100 / 8)
        velocity.append(data.iloc[i, 2])
        weight.append(data.iloc[i, 3])
        x.append(index)
        index = index + 1
