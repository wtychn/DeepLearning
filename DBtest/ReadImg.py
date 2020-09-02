import datetime
import io
import numpy as np
import pyodbc
from PIL import Image
import matplotlib.pyplot as plt

# 读取数据库
driver = 'SQL Server Native Client 11.0'  # 因版本不同而异
server = 'csu-tech.mynatapp.cc,65141'
user = 'sa'
password = '0000'
database = 'IronWData'

conn = pyodbc.connect(driver=driver, server=server, user=user, password=password, database=database)

cur = conn.cursor()
sql = "SELECT TOP 950 * FROM IronWData.dbo.TapholeState where 铁口号 = 1 and 铁口状态 = '开铁口' ORDER BY 时间 desc"  # 查询语句
cur.execute(sql)
rows = cur.fetchall()  # list
conn.close()

res = np.empty([950, 300, 930], dtype=int)

for i in range(950):
    # 读取数据库image格式和日期格式
    imgdata = rows[i][3]
    if imgdata is None:
        continue
    # date = rows[i][0]
    # day = datetime.datetime.strftime(date, '%Y%m%d%H%M%S')
    image = Image.open(io.BytesIO(imgdata))
    # image.show()
    # image.save('images/' + day + '.png')

    # 灰度化
    img = np.array(image)
    imgCut = img[500:800, 350:1280, :]
    b = np.array([0.299, 0.587, 0.114])
    imgGray = np.dot(imgCut, b)  # 将上面的RGB和b数组中的每个元素进行对位相乘，再相加，一定得到的是一个数字L
    # 二值化
    # rows, cols = imgGray.shape
    # imgBin = np.zeros([rows, cols])
    # for i in range(rows):
    #     for j in range(cols):
    #         if imgGray[i, j] <= 45:
    #             imgBin[i, j] = 0
    #         else:
    #             imgBin[i, j] = 1
    res[i] = imgGray
    # 显示图像
    # plt.subplot(3, 1, 1)
    # plt.imshow(imgGray, cmap="gray")
    # plt.subplot(3, 1, 2)
    # plt.imshow(imgBin, cmap="gray")
    # plt.subplot(3, 1, 3)
    # plt.imshow(imgBin, cmap="gray")
    # plt.show()

# 用日期命名
# np.save('images/' + day, imgBin)
np.save('images/image', res)
