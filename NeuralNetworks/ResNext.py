from __future__ import print_function, division

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm
from torchsummary import summary

batch_size = 16
learning_rate = 0.0002
Epoch = 100
useEpoch = 19
TRAIN = True
dataPath = '../DBtest/img/'
checkPointPath = '../DBtest/checkPoint/ResNeXt_checkPoint/'

train_transforms = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
test_transforms = transforms.Compose([
    # transforms.Resize(256),
    # transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
val_transforms = transforms.Compose([
    # transforms.Resize(256),
    # transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

train_dir = dataPath + 'train'
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

test_dir = dataPath + 'test'
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

val_dir = dataPath + 'val'
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True)


# --------------------模型定义---------------------------------
__all__ = ['ResNeXt', 'resnext_50', 'resnext_101', 'resnext_152']


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, num_classes=10, num_group=32):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_group)
        self.layer2 = self._make_layer(block, 128, layers[1], num_group, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], num_group, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext_50(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext_101(**kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model = ResNeXt(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext_152(**kwargs):
    """Constructs a ResNeXt-152 model.
    """
    model = ResNeXt(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


model = resnext_50()
if torch.cuda.is_available():
    model.cuda()
    print('Using GPU')
summary(model, (3, 224, 224))

# --------------------训练过程---------------------------------
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()
start_epoch = 0

if os.path.exists(checkPointPath + 'ResNeXt_best_' + str(useEpoch) + '.pth'):
    path_checkpoint = checkPointPath + 'ResNeXt_best_' + str(useEpoch) + '.pth'  # 断点路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch'] + 1  # 设置开始的epoch

if TRAIN:
    if os.path.exists(checkPointPath + 'loss.npy'):
        Loss_list = np.load(checkPointPath + 'loss.npy', allow_pickle=True)
    else:
        Loss_list = []

    if os.path.exists(checkPointPath + 'accuracy.npy'):
        Accuracy_list = np.load(checkPointPath + 'accuracy.npy', allow_pickle=True)
    else:
        Accuracy_list = []

    for epoch in range(start_epoch, Epoch):
        print('epoch {}'.format(epoch))
        # training-----------------------------
        model.train()
        train_loss = 0.
        train_acc = 0.
        print('Training...')
        trainBar = tqdm(total=len(train_dataloader))
        for step, (batch_x, batch_y) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.data
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainBar.update(1)
        trainBar.close()
        print(
            'Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / len(train_datasets), train_acc / len(train_datasets)))

        # 断点保存
        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, checkPointPath + 'ResNeXt_best_%s.pth' % (str(epoch)))

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        print('Testing...')
        testBar = tqdm(total=len(val_dataloader))
        for batch_x, batch_y in val_dataloader:
            with torch.no_grad():
                if torch.cuda.is_available():
                    batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
                else:
                    batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.data
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.data
            testBar.update(1)
        testBar.close()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / len(val_datasets), eval_acc / len(val_datasets)))

        # 断点保存
        if epoch >= len(Loss_list):
            Loss_list = np.hstack((Loss_list, eval_loss.data.cpu() / len(val_datasets)))
            Accuracy_list = np.hstack((Accuracy_list, 100 * eval_acc.data.cpu() / len(val_datasets)))
        else:
            Loss_list[epoch] = eval_loss.data.cpu() / len(val_datasets)
            Accuracy_list[epoch] = 100 * eval_acc.data.cpu() / len(val_datasets)

        np.save(checkPointPath + 'loss', Loss_list)
        np.save(checkPointPath + 'accuracy', Accuracy_list)

    x1 = range(0, 10)
    x2 = range(0, 10)
    y1 = Accuracy_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    # plt.savefig("accuracy_loss.jpg")

else:
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    print('Testing...')
    testBar = tqdm(total=len(test_dataloader))
    for batch_x, batch_y in test_dataloader:
        with torch.no_grad():
            if torch.cuda.is_available():
                batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data
        testBar.update(1)
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / len(test_datasets), eval_acc / len(test_datasets)))
    testBar.close()
