from __future__ import print_function, division

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm
from torchsummary import summary

from coatnet import coatnet_0

batch_size = 16
learning_rate = 0.0001
Epoch = 100
useEpoch = 83
TRAIN = True
dataPath = 'E:/new_imgs'
checkPointPath = 'E:/check_point/new_coatnet'

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

train_dir = os.path.join(dataPath, 'train')
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

test_dir = os.path.join(dataPath, 'test')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

val_dir = os.path.join(dataPath, 'val')
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True)

# --------------------模型定义---------------------------------
model = coatnet_0()
if torch.cuda.is_available():
    model.cuda()
    print('Using GPU')
# summary(model, (3, 224, 224))

# --------------------训练过程---------------------------------

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()
start_epoch = 0

path_checkpoint = os.path.join(checkPointPath, 'coatnet_best_' + str(useEpoch) + '.pth')
if os.path.exists(path_checkpoint):
    checkpoint = torch.load(path_checkpoint)  # 加载断点
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch'] + 1  # 设置开始的epoch

if TRAIN:
    loss_path = os.path.join(checkPointPath, 'loss.npy')
    if os.path.exists(loss_path):
        Loss_list = np.load(loss_path, allow_pickle=True)
    else:
        Loss_list = []

    accuracy_path = os.path.join(checkPointPath, 'accuracy.npy')
    if os.path.exists(accuracy_path):
        Accuracy_list = np.load(accuracy_path, allow_pickle=True)
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
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / len(train_datasets), train_acc / len(train_datasets)))

        # 断点保存
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, os.path.join(checkPointPath, 'coatnet_best_%s.pth' % (str(epoch))))

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        print('Testing...')
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
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / len(val_datasets), eval_acc / len(val_datasets)))

        # 断点保存
        if epoch >= len(Loss_list):
            Loss_list = np.hstack((Loss_list, eval_loss.data.cpu() / len(val_datasets)))
            Accuracy_list = np.hstack((Accuracy_list, 100 * eval_acc.data.cpu() / len(val_datasets)))
        else:
            Loss_list[epoch] = eval_loss.data.cpu() / len(val_datasets)
            Accuracy_list[epoch] = 100 * eval_acc.data.cpu() / len(val_datasets)

        np.save(os.path.join(checkPointPath, 'loss'), Loss_list)
        np.save(os.path.join(checkPointPath, 'accuracy'), Accuracy_list)

    x1 = range(100)
    x2 = range(100)
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
    testBar.close()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / len(test_datasets), eval_acc / len(test_datasets)))
