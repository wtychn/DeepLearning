import matplotlib.pyplot as plt
import numpy as np

img = plt.imread('images/*.png')

imgCut = img[500:800, 350:1000, :]

b = np.array([0.299, 0.587, 0.114])
imgGray = np.dot(imgCut, b)  # 将上面的RGB和b数组中的每个元素进行对位相乘，再相加，一定得到的是一个数字L

rows, cols = imgGray.shape
for i in range(rows):
    for j in range(cols):
        if imgGray[i, j] <= 0.6:
            imgGray[i, j] = 0
        else:
            imgGray[i, j] = 1

plt.imshow(imgGray, cmap="gray")
plt.show()
