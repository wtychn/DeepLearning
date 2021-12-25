from __future__ import print_function, division

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm
from torchsummary import summary
from dataset import TestImageDataset

from coatnet import coatnet_0
from DBtest.predicted import Predict
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

batch_size = 16
learning_rate = 0.0001
Epoch = 100
useEpoch = 83
TRAIN = False
dataPath = 'E:\\data'
checkPointPath = '../../DBtest/checkPoint/coatnet_checkPoint/'

test_dirs = [3, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21]
for num in test_dirs:
    test_dir = os.path.join(dataPath, str(num))
    test_datasets = TestImageDataset(test_dir, (224, 224), (224, 224))
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False)
    # --------------------模型定义---------------------------------
    model = coatnet_0()
    if torch.cuda.is_available():
        model.cuda()
        print('Using GPU')
    # summary(model, (3, 224, 224))

    # --------------------测试过程---------------------------------

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    start_epoch = 0

    if os.path.exists(checkPointPath + 'coatnet_best_' + str(useEpoch) + '.pth'):
        path_checkpoint = checkPointPath + 'coatnet_best_' + str(useEpoch) + '.pth'  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch'] + 1  # 设置开始的epoch

    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    print('Testing...')
    testBar = tqdm(total=len(test_dataloader))
    classes = []
    predict = Predict()
    rests = []
    pros = []
    for _, (batch_x, batch_y, times) in enumerate(test_dataloader):
        with torch.no_grad():
            if torch.cuda.is_available():
                batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data
        pred = torch.max(out, 1)[1]

        rest, probability = predict.updates(pred.data.cpu().numpy(), times.data.cpu().numpy())
        classes = np.concatenate([classes, pred.data.cpu().numpy()])
        rests = np.concatenate([rests, np.array(rest)])
        pros = np.concatenate([pros, np.array(probability)])

        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data
        testBar.update(1)
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / len(test_datasets), eval_acc / len(test_datasets)))
    testBar.close()

    # ------------------计算剩余时间真值----------------------------
    rest_times = []
    filenames = os.listdir(test_dir)
    end_time = datetime.strptime(filenames[len(filenames) - 1], '%Y_%m_%d_%H_%M_%S.jpg')
    for filename in filenames:
        cur_time = datetime.strptime(filename, '%Y_%m_%d_%H_%M_%S.jpg')
        rest_time = (end_time - cur_time).seconds / 60
        rest_times.append(rest_time)

    rest_times = rest_times[(len(rest_times) - len(rests)):]

    nprests = abs(np.array(rest_times) - np.array(rests))
    nprests_mean = np.mean(nprests)
    print(str(num) + ' Test Loss: {:.6f}'.format(nprests_mean))

    slopes = []
    for i in range(1, len(rests)):
        slopes.append((rests[i] - rests[i - 1]))

    # ------------------------绘图---------------------------------
    # plt.plot(np.arange(0, len(classes)), classes, '-', color='red')
    # plt.title('Test classes vs. images')
    # plt.ylabel('Test classes')
    # plt.show()

    plt.plot(np.arange(0, len(rests)), rests, '-', color='green', label='predict')
    plt.plot(np.arange(0, len(rests)), rest_times, '-', color='blue', label='true')
    plt.title('Test rest time vs. images')
    plt.ylabel('Test rest time')
    plt.legend()
    save_path = os.path.join('E:/check_point/predict/new31', str(num) + '_' + str(nprests_mean) + '_' + '.jpg')
    plt.savefig(save_path)
    plt.show()

    # plt.plot(np.arange(0, len(slopes)), slopes, '-', color='red')
    # plt.title('Test slope vs. images')
    # plt.ylabel('Test slope')
    # plt.show()

    # plt.plot(np.arange(0, len(pros)), pros, '-', color='blue')
    # plt.title('Test probability vs. images')
    # plt.ylabel('Test probability')
    # plt.show()
