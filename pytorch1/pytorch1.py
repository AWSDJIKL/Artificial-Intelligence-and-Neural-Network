import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_data():
    x1 = np.load('x1_2.npy')
    x2 = np.load('x2_2.npy')
    x = np.zeros((2, x1.shape[0]))
    x[0, :] = x1
    x[1, :] = x2
    y0 = np.load('y_2.npy')
    y = y0
    xtest = x[:, -100:]
    ytest = y[-100:]
    x_train = x[:, :-100]
    y_train = y[:-100]
    return x_train, y_train, xtest, ytest


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


if __name__ == '__main__':
    net = Net()
    x_train, y_train, xtest, ytest = load_data()
    lr = 0.1
    loopnum = 50
    myrange = list(range(y_train.shape[0]))
    for i in range(loopnum):
        print('loop=%d' % i)
        random.shuffle(myrange)
        for j in myrange:
            xt = torch.tensor(x_train[:, j], dtype=torch.float32)
            yt = torch.tensor(y_train[j], dtype=torch.float32)
            #梯度重置
            net.zero_grad()
            #计算预测值
            ypred = net(xt)
            #计算损失
            criterion = nn.MSELoss()
            loss = criterion(ypred, yt)
            #反向传播
            loss.backward()
            for f in net.parameters():
                f.data = f.data - f.grad.data * lr
    ypredlist = []
    for j in range(ytest.shape[0]):
        xt = torch.tensor(xtest[:, j], dtype=torch.float32)
        yt = torch.tensor(y_train[j], dtype=torch.float32)
        ypred = net(xt)
        ypredlist.append(np.array(ypred.data))
    ypredlist = np.array(ypredlist)
    ypredlist = ypredlist.reshape(100)
    MSE = np.sum((ytest - ypredlist) ** 2) / ytest.shape[0]
    plt.figure(figsize=(10, 5))
    plt.plot(ytest, 'r+', label='real data')
    plt.plot(ypredlist, 'b*', label='pred data')
    plt.legend()
    plt.grid()
    plt.title('MSE=%5.2f' % MSE)
    plt.savefig('out.jpg', dpi=256)
    plt.close()
