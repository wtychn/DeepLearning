import os
import shutil
import random

from tqdm import tqdm
from ImgTreatment import ImgTreatment


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


count = 0
page = 0
for page in range(10):
    src_path = os.path.join('E:/new_imgs/train', str(page))
    filenames = random.sample(os.listdir(src_path), 100)
    pbar = tqdm(total=len(filenames))
    for filename in filenames:
        pbar.update(1)
        target_path = os.path.join('E:/new_imgs/val', str(page))
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        full_path = os.path.join(src_path, filename)
        shutil.move(full_path, target_path)
    pbar.close()
