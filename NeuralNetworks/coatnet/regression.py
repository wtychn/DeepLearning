from __future__ import print_function, division

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm

from tensorboardX import SummaryWriter

from coatnet import coatnet_r
from dataset import RegressionImageDataset

batch_size = 16
learning_rate = 0.0001
Epoch = 100
useEpoch = 83
TRAIN = True
train_test_val = [.8, .1, .1]
dataPath = 'E:/data'
checkPointPath = 'E:/check_point/regression'
tensorboard_path = os.path.join(checkPointPath, 'tensorborad_log')
npy_path = os.path.join(checkPointPath, 'npy')

train_transforms = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

src_datasets = RegressionImageDataset(dataPath, (224, 224), (224, 224), train_transforms)

train_size = int(len(src_datasets) * train_test_val[0])
val_size = int(len(src_datasets) * train_test_val[2])
test_size = len(src_datasets) - val_size - train_size
train_datasets, test_datasets, val_datasets = torch.utils.data.random_split(
    src_datasets,
    [train_size, test_size, val_size]
)

train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True)

# --------------------模型定义---------------------------------
model = coatnet_r()
if torch.cuda.is_available():
    model.cuda()
    print('Using GPU')
# summary(model, (3, 224, 224))

# --------------------训练过程---------------------------------

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()
start_epoch = 0

path_checkpoint = os.path.join(checkPointPath, 'coatnet_best_' + str(useEpoch) + '.pth')
if os.path.exists(path_checkpoint):
    checkpoint = torch.load(path_checkpoint)  # 加载断点
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch'] + 1  # 设置开始的epoch

if TRAIN:
    loss_path = os.path.join(npy_path, 'loss.npy')
    if os.path.exists(loss_path):
        Loss_list = np.load(loss_path, allow_pickle=True)
    else:
        Loss_list = []

    accuracy_path = os.path.join(npy_path, 'accuracy.npy')
    if os.path.exists(accuracy_path):
        Accuracy_list = np.load(accuracy_path, allow_pickle=True)
    else:
        Accuracy_list = []

    writer = SummaryWriter(tensorboard_path)

    count = 0
    for epoch in range(start_epoch, Epoch):
        print('epoch {}'.format(epoch))
        # training-----------------------------
        model.train()
        train_loss = 0.
        train_acc = 0.
        print('Training...')
        trainBar = tqdm(total=len(train_dataloader))
        for batch_x, batch_y in train_dataloader:
            count += 1
            if torch.cuda.is_available():
                batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda().float()
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.data
            train_correct = torch.abs(out - batch_y).sum()
            train_acc += train_correct.data

            writer.add_scalar('train loss', loss.item(), count)
            writer.add_scalar('train accuracy', torch.abs(out - batch_y).sum(), count)

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
            num_correct = torch.abs(out - batch_y).sum()
            eval_acc += num_correct.data
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / len(val_datasets), eval_acc / len(val_datasets)))

        # 断点保存
        if epoch >= len(Loss_list):
            Loss_list = np.hstack((Loss_list, eval_loss.data.cpu() / len(val_datasets)))
            Accuracy_list = np.hstack((Accuracy_list, eval_acc.data.cpu() / len(val_datasets)))
        else:
            Loss_list[epoch] = eval_loss.data.cpu() / len(val_datasets)
            Accuracy_list[epoch] = eval_acc.data.cpu() / len(val_datasets)

        np.save(os.path.join(npy_path, 'loss'), Loss_list)
        np.save(os.path.join(npy_path, 'accuracy'), Accuracy_list)

    writer.close()

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
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / len(test_datasets), eval_acc / len(test_datasets)))
    testBar.close()
