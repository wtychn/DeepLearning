# library
# standard library

import numpy as np
import matplotlib.pyplot as plt
# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 40
LR = 0.001  # learning rate


def pil_loader(path):  # 一般采用pil_loader函数。
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


dataset = dset.ImageFolder('../DBtest/img/data/train', loader=pil_loader, transform=transforms.ToTensor())

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# pick 2000 samples to speed up testing
test_data = dset.ImageFolder('../DBtest/img/data/test', loader=pil_loader, transform=transforms.ToTensor())
test_loader = Data.DataLoader(dataset=test_data, batch_size=len(dataset.samples), shuffle=True)
for i, (imgs, targets) in enumerate(test_loader):
    test_x = imgs
    test_y = targets
test_x = test_x[:20].cuda()
test_y = test_y[:20].cuda()
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


cnn = CNN()
cnn.cuda()
# print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# following function (plot_with_labels) is for visualization, can be ignored if not interested
# from matplotlib import cm
#
# try:
#     from sklearn.manifold import TSNE
#
#     HAS_SK = True
# except:
#     HAS_SK = False
#     print('Please install sklearn for layer visualization')
#
#
# def plot_with_labels(lowDWeights, labels):
#     plt.cla()
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         c = cm.rainbow(int(255 * s / 9));
#         plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#     plt.xlim(X.min(), X.max());
#     plt.ylim(Y.min(), Y.max());
#     plt.title('Visualize last layer');
#     plt.show();
#     plt.pause(0.01)


# plt.ion()
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
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
            # if HAS_SK:
            #     # Visualization of trained flatten layer (T-SNE)
            #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            #     plot_only = 500
            #     low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
            #     labels = test_y.numpy()[:plot_only]
            #     plot_with_labels(low_dim_embs, labels)
# plt.ioff()

# print 10 predictions from test data
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].cuda().data
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
