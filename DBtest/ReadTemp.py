import io
import numpy as np
import pyodbc

# 读取数据库
driver = 'SQL Server Native Client 11.0'  # 因版本不同而异
server = 'csu-tech.mynatapp.cc,65141'
user = 'sa'
password = '0000'
database = 'IronWData'

conn = pyodbc.connect(driver=driver, server=server, user=user, password=password, database=database)

cur = conn.cursor()
sql = "SELECT 时间,铁水红外温度 FROM IronWData.dbo.IronTemp where 铁口号 = 1 and 时间 > '2020-03-03' ORDER BY 时间 desc"  # 查询语句
cur.execute(sql)
rows = cur.fetchall()  # list
conn.close()

np.save("temp", rows)
