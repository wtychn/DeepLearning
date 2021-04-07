import os
import shutil

for i in range(1, 18):
    path = 'D:\\OneDrive - csu.edu.cn\\研究课题\\铁水图像视频\\molten iron at taphole' + str(
        i) + '\\molten iron at taphole' + str(i)
    filenames = os.listdir(path)
    n = len(filenames)
    pageCap = int(n / 10) + 1
    page = 0
    # while page < 10:
    for j in range(page * pageCap, (page + 1) * pageCap):
        if j >= len(filenames):
            break
        full_path = os.path.join(path, filenames[j])
        new_dir = 'D:\\code\\python\\DeepLearning\\DBtest\\img\\train\\' + str(page)
        new_file = os.path.join(new_dir, filenames[j])
        shutil.copy(full_path, new_file)
        # page += 1
