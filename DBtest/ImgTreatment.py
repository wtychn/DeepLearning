import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


class ImgTreatment:

    def imgRepair(self, src):
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
                if screenNum[i] > 0 and screenNum[i] < 255:
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

    def treatment(self, origin_dir, target_dir, isRepair=True, isDehaze=True, isResize=True, isShow=False,
                  isSave=False):
        files = os.listdir(origin_dir)
        pbar = tqdm(total=len(files))
        for filename in files:
            all_path = os.path.join(origin_dir, filename)
            target_path = os.path.join(target_dir, filename)
            img = cv2.imread(all_path)  # 图片读取
            repair = img
            m = img
            resize = img
            # 图像修复
            if isRepair:
                repair = self.imgRepair(img)
            # 图像去雾(自动色阶去雾算法)
            if isDehaze:
                m = self.CreateNewImg(repair)
            # 改变图片尺寸
            if isResize:
                resize = self.resize(m)

            if isSave:
                cv2.imwrite(target_path, resize)

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
            pbar.update(1)

        # cv2.namedWindow("img", 0)
        # cv2.resizeWindow("img", int(width / 2), int(hight / 2))
        # cv2.imshow('img', img)
        #
        # cv2.namedWindow("newImage", 0)
        # cv2.resizeWindow("newImage", int(width / 2), int(hight / 2))
        # cv2.imshow("newImage", m)
        # cv2.waitKey(0)
