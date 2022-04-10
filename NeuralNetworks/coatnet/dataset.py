import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_image_file(filename):
    # ------------------------------------------------------
    # https://www.runoob.com/python/python-func-any.html
    # https://www.runoob.com/python/att-string-endswith.html
    # ---------------------------------------------------
    return any([filename.endswith(extention) for extention in
                ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']])


def image_transforms():
    # --------------------------------------------------------------
    # https://blog.csdn.net/weixin_38533896/article/details/86028509
    # --------------------------------------------------------------
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])


class TestImageDataset(Dataset):

    def __init__(self, image_root, load_size, crop_size):
        super(TestImageDataset, self).__init__()

        # self.image_files = [os.path.join(root, file) for root, dirs, files in os.walk(image_root)
        #                     for file in files if check_image_file(file)]
        self.image_files = []
        for filename in os.listdir(image_root):
            img_path = os.path.join(image_root, filename)
            if check_image_file(img_path):
                self.image_files.append(img_path)

        self.number_image = len(self.image_files)
        self.load_size = load_size
        self.crop_size = crop_size
        self.image_files_transforms = image_transforms()
        self.start = datetime.strptime(os.path.basename(self.image_files[0]), '%Y_%m_%d_%H_%M_%S.jpg')

    def __getitem__(self, index):

        image = Image.open(self.image_files[index % self.number_image])

        filename = os.path.basename(self.image_files[index % self.number_image])
        time = (datetime.strptime(filename, '%Y_%m_%d_%H_%M_%S.jpg') - self.start).seconds

        ground_truth = self.image_files_transforms(image.convert('RGB'))

        page_size = int(self.number_image / 10) + 1

        flag = int(index / page_size)

        return ground_truth, flag, time

    def __len__(self):
        return self.number_image


class RegressionImageDataset(Dataset):

    def __init__(self, image_root, load_size, crop_size, img_transforms):
        super(RegressionImageDataset, self).__init__()

        # self.image_files = [os.path.join(root, file) for root, dirs, files in os.walk(image_root)
        #                     for file in files if check_image_file(file)]
        self.image_files = []

        dir_paths = os.listdir(image_root)
        for dir in dir_paths:
            img_paths = os.listdir(os.path.join(image_root, dir))
            start = int(len(img_paths) * 0.9)
            for i in range(start, len(img_paths)):
                img_path = os.path.join(image_root, dir, img_paths[i])
                if check_image_file(img_path):
                    idx = (i - start) / (len(img_paths) - start) * 10
                    path = [img_path, idx]
                    self.image_files.append(path)

        self.number_image = len(self.image_files)
        self.load_size = load_size
        self.crop_size = crop_size
        self.image_files_transforms = img_transforms

    def __getitem__(self, index):
        paths = self.image_files[index % self.number_image]

        image = Image.open(paths[0])

        ground_truth = self.image_files_transforms(image.convert('RGB'))

        return ground_truth, paths[1]

    def __len__(self):
        return self.number_image


class FusionDataset(Dataset):

    def __init__(self, data_dir, img_dir):
        super(FusionDataset, self).__init__()

        dictionary_array = np.array([])
        flag = 0

        for filename in os.listdir(data_dir):
            data_path = os.path.join(data_dir, filename)
            df = pd.read_csv(data_path, header=None)
            for index, row in df.iterrows():
                img_path = os.path.join(os.path.join(img_dir, str(flag)), row[0] + ".jpg")
                if os.path.exists(img_path):
                    dictionary = {
                        "image": cv2.imread(img_path).reshape(3, 224, 224),
                        "x": row[2:].to_numpy(),
                        "y": flag
                    }
                    dictionary_array = np.append(dictionary_array, dictionary)
            flag += 1

            self.datas = dictionary_array

    def __getitem__(self, index):
        return self.datas[index]

    def __len__(self):
        return len(self.datas)
