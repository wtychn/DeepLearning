"""
提取glcm纹理特征
"""
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageStat
from PIL import Image
from skimage.feature import greycomatrix, greycoprops


def get_inputs(path):  # s为图像路径
    input = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 读取图像，灰度模式

    # 得到共生矩阵，参数：图像矩阵，距离，方向，灰度级别，是否对称，是否标准化
    glcm = greycomatrix(
        input,
        [16],  # [2, 8, 16]
        [np.pi / 2],  # [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4]
        256,
        symmetric=True,
        normed=True
    )

    # print(glcm)

    res = np.array([])
    # 得到共生矩阵统计值
    for prop in {'contrast', 'dissimilarity',
                 'homogeneity', 'energy', 'correlation', 'ASM'}:
        temp = greycoprops(glcm, prop)
        # temp=np.array(temp).reshape(-1)
        res = np.append(res, temp)

    state = ImageStat.Stat(Image.fromarray(input))
    res = np.append(res, state.mean[0])

    return res


if __name__ == '__main__':
    res = []
    res.append(get_inputs("../img/data/train/0/2018-03-26_17_45_46.jpg"))
    res.append(get_inputs("../img/data/train/0/2018-03-26_17_45_56.jpg"))
    res.append(get_inputs("../img/data/train/0/2018-03-26_17_46_06.jpg"))

    res.append(get_inputs("../img/data/train/1/2018-04-02_07_10_27.jpg"))
    res.append(get_inputs("../img/data/train/1/2018-04-02_07_10_57.jpg"))
    res.append(get_inputs("../img/data/train/1/2018-04-02_07_11_07.jpg"))

    df = pd.DataFrame(res, columns=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM', 'light'])

    print(df)
