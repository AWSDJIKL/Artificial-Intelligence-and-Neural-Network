# -*- coding: utf-8 -*-
'''
使用pytorch通用架构完成作业2，速度更快
'''
# @Time : 2021/4/7 14:31 
# @Author : LINYANZHEN
# @File : pytorch2_2_framework.py


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

# 定义数据集类，需要集成torch.utils.data.Dataset类，至少重写init,geiitem,len三个魔法方法
class MeantDataset(Dataset):
    def __init__(self, file_path):
        super(MeantDataset, self).__init__()
        # 只要第三列
        data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=2)
        x = np.zeros((data.shape[0] - 6, 6))
        y = np.zeros((data.shape[0] - 6, 1))
        for i in range(6):
            x[:, i] = data[i:-(6 - i)]
        y[:, 0] = data[6:]
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


# 定义训练数据集类
class MeantTrainDataset(Dataset):
    def __init__(self, file_path):
        super(MeantTrainDataset, self).__init__()
        # 只要第三列
        data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=2)
        x = np.zeros((33108 - 6, 6))
        y = np.zeros((33108 - 6, 1))
        for i in range(6):
            x[:, i] = data[i:33108 - (6 - i)]
        y[:, 0] = data[6:33108]
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


# 定义测试数据集类
class MeantTestDataset(Dataset):
    def __init__(self, file_path):
        super(MeantTestDataset, self).__init__()
        # 只要第三列
        data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=2)
        x = np.zeros((data.shape[0] - 33108 - 6, 6))
        y = np.zeros((data.shape[0] - 33108 - 6, 1))
        for i in range(6):
            x[:, i] = data[33108 + i: -(6 - i)]
        y[:, 0] = data[33108 + 6:]
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


# 定义模型，需要集成torch.nn.Module类，并重写init,forward，2个方法
class MeantModel(nn.Module):
    def __init__(self):
        super(MeantModel, self).__init__()
        # 2 input image channel, 1 output channels
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6, 4)
        self.fc2 = nn.Linear(4, 2)
        self.fc3 = nn.Linear(2, 1)
        self.sigmoid = torch.sigmoid

    def forward(self, x):
        # If the size is a square you can only specify a single number
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        # x = x.squeeze(-1)
        return x


# 训练
def train(model, train_loader, criterion, optimizer, epoch):
    for e in range(epoch):
        print("当前训练轮次：{}".format(e))
        for i, (x, y) in enumerate(train_loader, 0):
            # 计算预测值
            y_pred = model(x.float())
            # 梯度归零
            optimizer.zero_grad()
            # 计算损失，反向传播
            loss = criterion(y_pred.double(), y.double())
            loss.backward()
            # 更新权重
            optimizer.step()
    return


# 测试
def test(model, test_loader):
    y_pred_list = []
    y_true_list = []
    for i, (x, y) in enumerate(test_loader, 0):
        # 计算预测值
        y_pred = model(x.float())
        if i == 0:
            y_pred_list = y_pred.detach().numpy()
            y_true_list = y.detach().numpy()
        else:
            y_pred_list = np.concatenate((y_pred_list, y_pred.detach().numpy()))
            y_true_list = np.concatenate((y_true_list, y.detach().numpy()))

    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(y_true_list, 'r+', label='real data')
    plt.plot(y_pred_list, 'b*', label='pred data')
    plt.legend()
    plt.grid()
    MSE = np.sum((y_true_list - y_pred_list) ** 2) / y_true_list.shape[0]
    plt.title('MSE=%5.2f' % MSE)
    plt.savefig('out2_framework.jpg', dpi=256)
    plt.close()
    print(MSE)
    return


if __name__ == '__main__':
    start_time=time.time()
    model = MeantModel()
    train_set = MeantTrainDataset("meant.csv")
    test_set = MeantTestDataset("meant.csv")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    train_loader = DataLoader(dataset=train_set, batch_size=12, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_set, batch_size=12, shuffle=False, num_workers=8)
    train(model, train_loader, criterion, optimizer, epoch=200)
    test(model, test_loader)
    end_time = time.time()
    print("用时：{}min".format((end_time - start_time) / 60))