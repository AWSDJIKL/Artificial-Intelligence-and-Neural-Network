# -*- coding: utf-8 -*-
'''
作业1
'''
# @Time : 2021/4/1 17:09 
# @Author : LINYANZHEN
# @File : pytorch2_1.py

import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 读取数据集并分割
def load_data():
    x1 = np.load("x1_2.npy")
    x2 = np.load("x2_2.npy")
    y0 = np.load("y_2.npy")
    x = np.zeros((2, x1.shape[0]))
    x[0, :] = x1
    x[1, :] = x2
    y = y0
    x_test = x[:, -100:]
    y_test = y[-100:]
    x_train = x[:, :-100]
    y_train = y[:-100]
    return x_train, y_train, x_test, y_test


# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 2 input image channel, 1 output channels
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        # If the size is a square you can only specify a single number
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = x.squeeze(-1)
        return x


def train(net, x, y, lr, epouch, optimizer):
    myrange = list(range(y.shape[0]))
    for i in range(epouch):
        print('loop=%d' % i)
        random.shuffle(myrange)
        # 遍历整个训练集
        for j in myrange:
            # 提取x,y
            xt = torch.tensor(x[:, j], dtype=torch.float32)
            yt = torch.tensor(y[j], dtype=torch.float32)
            # 梯度归零
            net.zero_grad()
            # 计算预测y
            ypred = net(xt)
            # 计算损失
            criterion = nn.MSELoss()
            loss = criterion(ypred, yt)
            # backward
            # 反向传播更新参数
            loss.backward()
            optimizer.step()
            # for f in net.parameters():
            #     f.data = f.data - f.grad.data * lr


def test(x_test, y_test):
    ypredlist = []
    for j in range(y_test.shape[0]):
        xt = torch.tensor(x_test[:, j], dtype=torch.float32)
        ypred = net(xt)
        ypredlist.append(np.array(ypred.data))
    ypredlist = np.array(ypredlist)
    ypredlist = ypredlist.reshape(100)
    print(ypredlist)
    print(y_test)
    MSE = np.sum((y_test - ypredlist) ** 2) / y_test.shape[0]
    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, 'r+', label='real data')
    plt.plot(ypredlist, 'b*', label='pred data')
    plt.legend()
    plt.grid()
    plt.title('MSE=%5.2f' % MSE)
    plt.savefig('out1.jpg', dpi=256)
    plt.close()


if __name__ == '__main__':
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    x, y, x_test, y_test = load_data()
    ##train loop
    lr = 0.1
    epouch = 200
    train(net, x, y, lr, epouch, optimizer)
    test(x_test, y_test)
