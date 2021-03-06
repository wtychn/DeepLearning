# library
# standard library

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
import os

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 40
LR = 0.001  # learning rate
TRAIN = False


def pil_loader(path):  # 一般采用pil_loader函数。
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


dataset = dset.ImageFolder('../DBtest/img/data/train', loader=pil_loader, transform=transforms.ToTensor())

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# pick 2000 samples to speed up testing
test_data = dset.ImageFolder('../DBtest/img/data/test', loader=pil_loader, transform=transforms.ToTensor())
test_loader = Data.DataLoader(dataset=test_data, batch_size=len(test_data.samples), shuffle=True)
for i, (imgs, targets) in enumerate(test_loader):
    imgs_x = imgs
    targets_y = targets
test_x = imgs_x[100:120].cuda()
test_y = targets_y[100:120].cuda()
print('Data ready!')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 408, 1024)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 408, 1024)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 204, 512)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 204, 512)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 204, 512)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 102, 256)
        )
        self.conv3 = nn.Sequential(  # input shape (32, 102, 256)
            nn.Conv2d(32, 32, 5, 1, 2),  # output shape (32, 102, 256)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 51, 128)
        )
        self.conv4 = nn.Sequential(  # input shape (32, 51, 128)
            nn.Conv2d(32, 32, 5, 1, 2),  # output shape (32, 51, 128)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 25, 64)
        )
        self.classification = nn.Linear(32 * 25 * 64, 40)  # fully connected layer, output 20 classes
        self.out = nn.Linear(40, 2)  # fully connected layer, output 2 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.classification(x)
        output = self.out(x)
        return output  # return x for visualization



if TRAIN:  # 如果TRAIN为真则开始训练
    cnn = CNN()
    cnn.cuda()
    # print(cnn)  # net architecture

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    # training and testing
    for epoch in range(EPOCH):
        for step, (imgs, targets) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            b_x = imgs.cuda()
            b_y = targets.cuda()

            output = cnn(b_x)  # cnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 10 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].cuda().data
                accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(),
                      '| test accuracy: %.2f' % accuracy)

    # print 10 predictions from test data
    test_output = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].cuda().data
    print(pred_y, 'prediction number')
    print(test_y[:10], 'real number')

    torch.save(cnn.state_dict(), 'cnn_params.pkl')

elif os.path.isfile('cnn_params.pkl'):  # cnn参数文件存在则直接测试
    cnn = CNN()
    cnn.load_state_dict(torch.load('cnn_params.pkl'))
    cnn.cuda()

    total_accuracy = 0
    for i in range(0, len(test_data.samples) - 20, 20):
        test_x = imgs_x[i:i + 20].cuda()
        test_y = targets_y[i:i + 20].cuda()
        test_output = cnn(test_x)
        pred_y = torch.max(test_output, 1)[1].cuda().data
        accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
        total_accuracy += accuracy
    total_accuracy /= len(test_data.samples) / 20
    print('test accuracy: %.2f' % total_accuracy)

else:
    print('CNN need to be train!')
