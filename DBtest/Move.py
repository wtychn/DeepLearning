import os
import shutil
import random

from tqdm import tqdm


class Move:
    def copy(self, srcDir, tarDir):
        for i in range(1, 23):
            path = srcDir + '\\' + str(i)
            filenames = os.listdir(path)
            n = len(filenames)
            pageCap = int(n / 10) + 1
            page = 0
            while page < 10:
                for j in range(page * pageCap, (page + 1) * pageCap):
                    if j >= len(filenames):
                        break
                    full_path = os.path.join(path, filenames[j])
                    new_dir = tarDir + '\\' + str(page)
                    new_file = os.path.join(new_dir, filenames[j])
                    shutil.copy(full_path, new_file)
                page += 1

    def cut(self, fileDir, tarDir, proportion=0.3):
        pathDir = os.listdir(fileDir)  # 取图片的原始路径
        filenumber = len(pathDir)
        rate = proportion  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
        picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
        sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
        print(sample)
        for name in sample:
            shutil.copy(fileDir + name, tarDir + name)
        return


for i in range(10):
    path = os.path.join('D:\\code\\python\\DeepLearning\\DBtest\\img\\train', str(i))
    filenames = os.listdir(path)
    pbar = tqdm(total=len(filenames))
    for filename in filenames:
        pbar.update(1)
        if filename.startswith('2018'):
            continue
        full_path = os.path.join(path, filename)
        new_dir = 'D:\\code\\python\\PConv\\dataset\\train\\train'
        new_file = os.path.join(new_dir, str(i), filename)
        shutil.copy(full_path, new_file)
    pbar.close()
