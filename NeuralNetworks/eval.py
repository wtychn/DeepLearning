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
from coatnet.dataset import TestImageDataset

from VGG import VGG
from ResNet import ResNet
from resnext_model import resnext_50
import coatnet.coatnet as co
import coatnet.skcoatnet as sk

from DBtest.predicted import Predict
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

batch_size = 16
learning_rate = 0.0001
Epoch = 100
useEpoch = 99
TRAIN = False
dataPath = 'E:\\data'
vgg_checkPointPath = '../DBtest/checkPoint/VGG_checkPoint/vgg_best_149.pth'
resnet_checkPointPath = '../DBtest/checkPoint/ResNet_checkPoint/resnet_best_99.pth'
resnext_checkPointPath = '../DBtest/checkPoint/ResNeXt_checkPoint/resnext_best_99.pth'
coatnet_checkPointPath = '../DBtest/checkPoint/coatnet_checkPoint/coatnet_best_99.pth'
skcoatnet_checkPointPath = 'E:/check_point/SKCoatnet2_checkPoint/coatnet_best_99.pth'

# --------------------模型定义---------------------------------
vgg = VGG([2, 2, 3, 3, 3], num_classes=10)
resnet = ResNet()
resnext = resnext_50()
coatnet = co.coatnet_0()
skcoatnet = sk.coatnet_0()

if torch.cuda.is_available():
    vgg.cuda()
    resnet.cuda()
    resnext.cuda()
    coatnet.cuda()
    skcoatnet.cuda()
    print('Using GPU')
# summary(model, (3, 224, 224))

# --------------------模型初始化---------------------------------
checkPoints = [vgg_checkPointPath, resnet_checkPointPath, resnext_checkPointPath, coatnet_checkPointPath, skcoatnet_checkPointPath]
models = [vgg, resnet, resnext, coatnet, skcoatnet]
model_names = ['VGG', 'ResNet', 'ResNeXt', 'CoAtNet', 'SK-CoAtNet']
model_count = len(model_names)
optimizers = []
loss_func = nn.CrossEntropyLoss()

for i in range(5):
    optimizers.append(optim.Adam(models[i].parameters(), lr=learning_rate))

    if os.path.exists(checkPoints[i]):
        checkpoint = torch.load(checkPoints[i])  # 加载断点
        models[i].load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizers[i].load_state_dict(checkpoint['optimizer'])  # 加载优化器参数

    models[i].eval()

# --------------------测试过程---------------------------------
test_dirs = [3, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21]
# right_count = np.zeros((model_count, 10))
# all_count = np.zeros((model_count, 10))
# test_dirs = [19]

all_tar_res = []

for num in test_dirs:
    test_dir = os.path.join(dataPath, str(num))
    test_datasets = TestImageDataset(test_dir, (224, 224), (224, 224))
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False)
    all_classes = []
    all_predicts = []
    rest_times = []
    og_rest_times = []

    for i in range(model_count):
        eval_loss = 0.
        eval_acc = 0.
        print('Testing...')
        classes = []
        testBar = tqdm(total=len(test_dataloader))
        # predict = Predict()
        # rests = []
        # pros = []
        for _, (batch_x, batch_y, times) in enumerate(test_dataloader):
            with torch.no_grad():
                if torch.cuda.is_available():
                    batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
                else:
                    batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = models[i](batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.data
            pred = torch.max(out, 1)[1]

            pred_cpu = pred.data.cpu().numpy()
            y_cpu = batch_y.data.cpu().numpy()
            # for j in range(len(pred_cpu)):
            #     right_count[i, y_cpu[j]] += 1 if pred_cpu[j] == y_cpu[j] else 0
            #     all_count[i, y_cpu[j]] += 1

            # rest, probability = predict.updates(pred.data.cpu().numpy(), times.data.cpu().numpy())
            classes = np.concatenate([classes, pred_cpu])
            # rests = np.concatenate([rests, np.array(rest)])
            # pros = np.concatenate([pros, np.array(probability)])

            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.data
            testBar.update(1)
        testBar.close()
        print(model_names[i] + 'Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / len(test_datasets), eval_acc / len(test_datasets)))

        # ------------------计算剩余时间真值----------------------------
        # if i == 0:
        #     filenames = os.listdir(test_dir)
        #     start_time = datetime.strptime(filenames[0], '%Y_%m_%d_%H_%M_%S.jpg')
        #     end_time = datetime.strptime(filenames[len(filenames) - 1], '%Y_%m_%d_%H_%M_%S.jpg')
        #     all_time = (end_time - start_time).seconds
        #     rest_time = int(all_time / 10)
        #     rest_times.append(rest_time)
            # for filename in filenames:
            #     cur_time = datetime.strptime(filename, '%Y_%m_%d_%H_%M_%S.jpg')
            #     rest_time = (end_time - cur_time).seconds / 60
        #         og_rest_times.append(rest_time)
        #
        # rest_times = og_rest_times[(len(og_rest_times) - len(rests)):]
        # nprests = abs(np.array(rest_times) - np.array(rests))
        # nprests_mean = np.mean(nprests)
        # print(str(num) + ' MAE: {:.6f}'.format(nprests_mean))

        # slopes = []
        # for i in range(1, len(rests)):
        #     slopes.append((rests[i] - rests[i - 1]))

        # np_path = os.path.join('E:\\check_point\\predict\\results', 'vgg')
        # np_path = os.path.join('E:\\check_point\\predict\\results', 'resnet')
        # np_path = os.path.join('E:\\check_point\\predict\\results', 'resnext')
        # np_path = os.path.join('E:\\check_point\\predict\\results', 'coatnet')
        # np.save(np_path, rests)

        # all_predicts.append(rests)
        all_classes.append(classes)

    # ------------------------绘图---------------------------------
    # colors = ['lightsteelblue', 'yellowgreen', 'green', 'orange', 'red']
    # styles = ['--', '--', '--', '--', '-']
    class_count = int(len(all_classes[0]) / 10)
    true_classes = []
    true_class = 0
    count = 0
    for i in range(len(all_classes[0])):
        if count == class_count:
            count = 0
            if true_class < 9:
                true_class += 1
        true_classes.append(true_class)
        count += 1
    #
    # for i in range(5):
    #     plt.plot(np.arange(0, len(all_classes[i])), all_classes[i], styles[i], color=colors[i], label=model_names[i])
    # plt.plot(np.arange(0, len(true_classes)), true_classes, '-', color='blue', label='true')
    # plt.title('Classification Result vs. Remaining Time')
    # plt.ylabel('Classification Result')
    # plt.legend()
    # save_path = os.path.join('E:/check_point/predict/fusion_classes', str(num) + '_' + '.eps')
    # plt.savefig(save_path, format='eps')
    # plt.show()
    all_classes.append(true_classes)
    all_tar_res.append(all_classes)
    # for i in range(5):
    #     plt.plot(np.arange(0, len(all_predicts[i])), all_predicts[i], styles[i], color=colors[i], label=model_names[i])
    # plt.plot(np.arange(0, len(rest_times)), rest_times, '-', color='blue', label='true')
    # plt.title('Remaining Time vs. Image Index')
    # plt.ylabel('Remaining Time (min)')
    # plt.legend()
    # save_path = os.path.join('E:/check_point/predict/resttime', str(num) + '_' + '.jpg')
    # plt.savefig(save_path)
    # plt.show()

    # plt.plot(np.arange(0, len(slopes)), slopes, '-', color='red')
    # plt.title('Test slope vs. images')
    # plt.ylabel('Test slope')
    # plt.show()

    # plt.plot(np.arange(0, len(pros)), pros, '-', color='blue')
    # plt.title('Test probability vs. images')
    # plt.ylabel('Test probability')
    # plt.show()

np.save('E:\\check_point\\predict\\classes\\all_tar_res.npy', all_tar_res)
# np_savepath = 'E:/check_point/predict/numpy'
# np.save(os.path.join(np_savepath, 'right_count'), right_count)
# np.save(os.path.join(np_savepath, 'all_count'), all_count)
# accuracy_nparray = right_count / all_count
# print(accuracy_nparray)
