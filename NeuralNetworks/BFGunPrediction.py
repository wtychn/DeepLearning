import numpy as np
import torch
from torch import nn

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 100  # rnn时间点数量
INPUT_SIZE = 2  # rnn每个时间点输入的特征量 流速、温度
LR = 0.001  # learning rate

# 读取数据
data_x = np.load("../DBtest/DataProcess/data_x.npy")
data_y = np.load("../DBtest/DataProcess/data_c.npy")

train_x = data_x[:900]
train_y = data_y[:900]
test_x = torch.from_numpy(data_x[900:]).type(torch.FloatTensor).cuda()
test_y = torch.tensor(data_y[900:]).cuda()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,  # rnn hidden unit
            num_layers=3,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 11)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
rnn.cuda()
# print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step in range(train_x.shape[0]):
        s_x = train_x[step]
        s_y = torch.tensor([train_y[step]], dtype=torch.long).cuda()
        b_x = torch.from_numpy(s_x[np.newaxis, :]).type(torch.FloatTensor).cuda()  # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)  # rnn output
        loss = loss_func(output, s_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_x)  # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].cuda().data
            accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| train accuracy: %.2f' % accuracy)

# print 10 predictions from train data
test_output = rnn(test_x[:10].view(-1, 100, 2))
pred_y = torch.max(test_output, 1)[1].cuda().data
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
