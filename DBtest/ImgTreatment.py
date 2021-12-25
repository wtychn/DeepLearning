import os
import shutil

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


class ImgTreatment:

    def imgRepair(self, src):
        # 图片二值化处理，把[240, 240, 240]~[255, 255, 255]以外的颜色变成0
        thresh1 = cv2.inRange(src, np.array([180, 200, 200]), np.array([255, 255, 255]))
        thresh2 = cv2.inRange(src, np.array([0, 0, 0]), np.array([10, 10, 10]))
        thresh = cv2.add(thresh1, thresh2)

        # 创建形状和尺寸的结构元素
        kernel = np.ones((3, 3), np.uint8)

        # 扩张待修复区域
        hi_mask = cv2.dilate(thresh, kernel, iterations=1)
        specular = cv2.inpaint(src, hi_mask, 5, flags=cv2.INPAINT_TELEA)

        median = cv2.medianBlur(specular, 3)
        # median = specular
        return median

    def ComputeMinLevel(self, hist, pnum):
        index = np.add.accumulate(hist)
        return np.argwhere(index > pnum * 8.3 * 0.01)[0][0]

    def ComputeMaxLevel(self, hist, pnum):
        hist_0 = hist[::-1]
        Iter_sum = np.add.accumulate(hist_0)
        index = np.argwhere(Iter_sum > (pnum * 2.2 * 0.01))[0][0]
        return 255 - index

    def LinearMap(self, minlevel, maxlevel):
        if (minlevel >= maxlevel):
            return []
        else:
            index = np.array(list(range(256)))
            screenNum = np.where(index < minlevel, 0, index)
            screenNum = np.where(screenNum > maxlevel, 255, screenNum)
            for i in range(len(screenNum)):
                if 0 < screenNum[i] < 255:
                    screenNum[i] = (i - minlevel) / (maxlevel - minlevel) * 255
            return screenNum

    def CreateNewImg(self, img):
        h, w, d = img.shape
        newimg = np.zeros([h, w, d])
        for i in range(d):
            imghist = np.bincount(img[:, :, i].reshape(1, -1)[0])
            minlevel = self.ComputeMinLevel(imghist, h * w)
            maxlevel = self.ComputeMaxLevel(imghist, h * w)
            screenNum = self.LinearMap(minlevel, maxlevel)
            if (screenNum.size == 0):
                continue
            for j in range(h):
                newimg[j, :, i] = screenNum[img[j, :, i]]
        return newimg

    def resize(self, image):
        image = np.array(image, dtype='uint8')
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        width, height = image.size
        new_image_length = max(width, height)  # 获取新图边长
        new_image = Image.new("RGB", (new_image_length, new_image_length), (0, 0, 0))  # 生成一张正方形底图
        pad_len = int(abs(width - height) / 2)  # 预填充区域的大小
        box = (0, pad_len) if width > height else (pad_len, 0)
        new_image.paste(image, box)
        resize = new_image.resize((224, 224), Image.ANTIALIAS)
        return cv2.cvtColor(np.asarray(resize), cv2.COLOR_RGB2BGR)

    def treatment(self, origin_dir, target_dir, isRepair=True, isDehaze=True, isDenoise=False, isResize=True, isShow=False,
                  isSave=False):
        all_path = origin_dir
        target_path = target_dir
        img = cv2.imread(all_path)  # 图片读取
        repair = img
        m = img
        resize = img
        dst = img
        # 图像修复
        if isRepair:
            img = self.imgRepair(img)
            repair = img
        # 图像去雾(自动色阶去雾算法)
        if isDehaze:
            img = np.uint8(self.CreateNewImg(img))
            m = img
        # 图片滤波
        if isDenoise:
            # 均值滤波
            img = cv2.blur(img, (5, 5))
            # 高斯滤波
            img = cv2.GaussianBlur(img, (5, 5), 0)
            # 中值滤波
            img = cv2.medianBlur(img, 5)
            # 双边滤波
            img = cv2.bilateralFilter(img, 9, 75, 75)
            dst = img
        # 改变图片尺寸
        if isResize:
            img = self.resize(img)
            resize = img

        if isSave:
            cv2.imwrite(target_path, img)

        if isShow:
            plt.subplot(2, 2, 1)
            plt.title('src')
            plt.imshow(img[:, :, ::-1])
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2, 2, 2)
            plt.title('repaired')
            plt.imshow(repair[:, :, ::-1])
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2, 2, 3)
            plt.title('dehazed')
            plt.imshow(m[:, :, ::-1] / 255)
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2, 2, 4)
            plt.title('resized')
            plt.imshow(resize[:, :, ::-1])
            plt.xticks([])
            plt.yticks([])

            plt.show()


        # cv2.namedWindow("img", 0)
        # cv2.resizeWindow("img", int(width / 2), int(hight / 2))
        # cv2.imshow('img', img)
        #
        # cv2.namedWindow("newImage", 0)
        # cv2.resizeWindow("newImage", int(width / 2), int(hight / 2))
        # cv2.imshow("newImage", m)
        # cv2.waitKey(0)

# src_path = 'D:\\Data\\OneDrive - csu.edu.cn\\study\\img\\part1'
# tar_path = 'E:\\data'
# for i in range(1, 23):
#     img_path = os.path.join(src_path, str(i))
#     filenames = os.listdir(img_path)
#     new_dir = os.path.join(tar_path, str(i))
#     if not os.path.exists(new_dir):
#         os.makedirs(new_dir)
#     pbar = tqdm(total=len(filenames))
#     for filename in filenames:
#         origin_path = os.path.join(img_path, filename)
#         target_path = os.path.join(new_dir, filename)
#         img_treatment = ImgTreatment()
#         img_treatment.treatment(origin_path, target_path, isRepair=False, isDehaze=False, isSave=True)
#         pbar.update(1)
#     pbar.close()


