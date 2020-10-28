import os
import shutil

for i in range(1, 18):
    path = 'E:\\研究课题\\铁水图像视频\\molten iron at taphole' + str(i) + '\\molten iron at taphole' + str(i)
    filenames = os.listdir(path)
    # 取出前180个图像（开始半小时）
    for j in range(170):
        full_path = os.path.join(path, filenames[j])
        new_dir = 'D:\\code\\DeepLearning\\python\\Learning\\DBtest\\img\\data\\0'
        new_file = os.path.join(new_dir, filenames[j])
        shutil.copy(full_path, new_file)
    # 取出最后180个图像（最后半小时）
    for j in range(len(filenames) - 180, len(filenames) - 10):
        full_path = os.path.join(path, filenames[j])
        new_dir = 'D:\\code\\DeepLearning\\python\\Learning\\DBtest\\img\\data\\1'
        new_file = os.path.join(new_dir, filenames[j])
        shutil.copy(full_path, new_file)
