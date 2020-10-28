import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm


def imgRepair(src):
    # 图片二值化处理，把[240, 240, 240]~[255, 255, 255]以外的颜色变成0
    thresh1 = cv2.inRange(src, np.array([180, 200, 200]), np.array([255, 255, 255]))
    thresh2 = cv2.inRange(src, np.array([0, 0, 0]), np.array([10, 10, 150]))
    thresh = cv2.add(thresh1, thresh2)

    # 创建形状和尺寸的结构元素
    kernel = np.ones((3, 3), np.uint8)

    # 扩张待修复区域
    hi_mask = cv2.dilate(thresh, kernel, iterations=1)
    specular = cv2.inpaint(src, hi_mask, 5, flags=cv2.INPAINT_TELEA)

    median = cv2.medianBlur(specular, 3)
    # median = specular
    return median


def zmMinFilterGray(src, r=5):
    """最小值滤波，r是滤波器半径"""
    '''if r <= 0:
        return src
    h, w = src.shape[:2]
    I = src
    res = np.minimum(I  , I[[0]+range(h-1)  , :])
    res = np.minimum(res, I[range(1,h)+[h-1], :])
    I = res
    res = np.minimum(I  , I[:, [0]+range(w-1)])
    res = np.minimum(res, I[:, range(1,w)+[w-1]])
    return zmMinFilterGray(res, r-1)'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))  # 使用opencv的erode函数更高效


def guidedfilter(I, p, r, eps):
    """引导滤波，直接参考网上的matlab代码"""
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def getV1(m, r, eps, w, maxV1):  # 输入rgb图像，值范围[0,1]
    """计算大气遮罩图像V1和光照值A, V1 = 1-t/A"""
    V1 = np.min(m, 2)  # 得到暗通道图像
    V1 = guidedfilter(V1, zmMinFilterGray(V1, 7), r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()

    V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制

    return V1, A


def deHaze(src, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(src.shape)
    V1, A = getV1(src, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
    for k in range(3):
        Y[:, :, k] = (src[:, :, k] - V1) / (1 - V1 / A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
    return Y


if __name__ == '__main__':
    oringin_dir = 'img\\data\\1'
    target_dir = 'img\\data\\train\\1'
    files = os.listdir(oringin_dir)
    pbar = tqdm(total=len(files))
    for filename in files:
        all_path = os.path.join(oringin_dir, filename)
        target_path = os.path.join(target_dir, filename)
        img = cv2.imread(all_path)  # 图片读取
        imgCut = img[140:548, 0:1024]  # 图片裁剪
        hight, width, depth = imgCut.shape[0:3]
        # 图像修复
        repair = imgRepair(imgCut)
        # 图像去雾(暗通道去雾算法 效果一般)
        m = deHaze(repair / 255.0) * 255
        cv2.imwrite(target_path, m)
        # 灰度化
        deHazeImg = cv2.imread(target_path)
        gray = cv2.cvtColor(deHazeImg, cv2.COLOR_BGR2GRAY)
        # 对比度增强(限制对比度自适应直方图均衡化)
        clahe = cv2.createCLAHE(2, (8, 8))
        dst = clahe.apply(gray)
        cv2.imwrite(target_path, dst)
        pbar.update(1)
        # cv2.imshow('gray', gray)
        # cv2.imwrite('gray.jpg', gray)
        # cv2.imshow('dst', dst)
        # cv2.imwrite('dst.jpg', dst)
        # cv2.waitKey(0)

    # cv2.namedWindow("img", 0)
    # cv2.resizeWindow("img", int(width / 2), int(hight / 2))
    # cv2.imshow('img', img)
    #
    # cv2.namedWindow("newImage", 0)
    # cv2.resizeWindow("newImage", int(width / 2), int(hight / 2))
    # cv2.imshow("newImage", m)
    # cv2.waitKey(0)
