# -*- coding: utf-8 -*-
'''
作业2
'''
# @Time : 2021/4/5 18:40 
# @Author : LINYANZHEN
# @File : pytorch2_2.py
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time


# 读取数据集并分割
def load_data():
    data = np.loadtxt("meant.csv", delimiter=",", skiprows=1, usecols=2)
    x = np.zeros((data.shape[0] - 6, 6))
    y = np.zeros((data.shape[0] - 6, 1))
    for i in range(6):
        x[:, i] = data[i:-(6 - i)]
    y[:, 0] = data[6:]
    x_train = x[:33108]
    y_train = y[:33108]
    x_test = x[33108:]
    y_test = y[33108:]
    # print(x_train)
    return x_train, y_train, x_test, y_test


# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 2 input image channel, 1 output channels
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6, 4)
        self.fc2 = nn.Linear(4, 2)
        self.fc3 = nn.Linear(2, 1)

    def forward(self, x):
        # If the size is a square you can only specify a single number
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def train(net, x, y, lr, epouch):
    myrange = list(range(y.shape[0]))
    for i in range(epouch):
        print('loop=%d' % i)
        random.shuffle(myrange)
        # 遍历整个训练集
        for j in myrange:
            # 提取x,y
            x_ture = torch.tensor(x[j], dtype=torch.float32)
            y_ture = torch.tensor(y[j], dtype=torch.float32)
            # 梯度归零
            net.zero_grad()
            # 计算预测y
            y_pred = net(x_ture)
            # 计算损失
            criterion = nn.MSELoss()
            loss = criterion(y_pred, y_ture)
            # backward
            # 反向传播更新参数
            loss.backward()
            for f in net.parameters():
                f.data = f.data - f.grad.data * lr


def test(x_test, y_test):
    ypredlist = []
    for j in range(y_test.shape[0]):
        xt = torch.tensor(x_test[j], dtype=torch.float32)
        ypred = net(xt)
        ypredlist.append(np.array(ypred.data))
    ypredlist = np.array(ypredlist)
    ypredlist = ypredlist.reshape(y_test.shape[0])
    MSE = np.sum((y_test - ypredlist) ** 2) / y_test.shape[0]
    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, 'r+', label='real data')
    plt.plot(ypredlist, 'b*', label='pred data')
    plt.legend()
    plt.grid()
    plt.title('MSE=%5.2f' % MSE)
    plt.savefig('out2.jpg', dpi=256)
    plt.close()
    print(MSE)


if __name__ == '__main__':
    start_time = time.time()
    net = Net()
    x_train, y_train, x_test, y_test = load_data()
    ##train loop
    lr = 0.01
    epouch = 200
    train(net, x_train, y_train, lr, epouch)
    test(x_test, y_test)
    end_time = time.time()
    print("用时：{}min".format((end_time - start_time) / 60))
