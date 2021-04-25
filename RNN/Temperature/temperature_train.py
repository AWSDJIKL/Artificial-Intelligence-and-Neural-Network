# -*- coding: utf-8 -*-
'''
循环神经网络预测温度模型训练
'''
# @Time : 2021/4/22 16:33 
# @Author : LINYANZHEN
# @File : temperature_train.py
import time

import torch
import torch.nn as nn
import temperature_model as tm
import temperature_dataset as td
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np


def train(model, train_loader, criterion, optimizer, epoch):
    loss_list = []
    for i in range(epoch):
        epoch_start_time = time.time()
        print("epoch[{}/{}]".format(i, epoch))
        for index, (x, y) in enumerate(train_loader, 0):
            out = model(x)
            optimizer.zero_grad()
            # out = out.view(17, 64)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print("  用时:{}min".format((time.time() - epoch_start_time) / 60))
    # 画出损失图像
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, "b")
    plt.title('loss')
    plt.savefig('Temperature_Train_loss.jpg', dpi=256)
    plt.close()


def test(model, test_loader):
    y_pred_list = []
    y_true_list = []
    for i, (x, y) in enumerate(test_loader, 0):
        # 计算预测值
        y_pred = model(x)
        y_pred = y_pred.detach().numpy().reshape(y_pred.shape[0])
        y = y.detach().numpy().reshape(y.shape[0])
        if i == 0:
            y_pred_list = y_pred
            y_true_list = y
        else:
            y_pred_list = np.concatenate((y_pred_list, y_pred))
            y_true_list = np.concatenate((y_true_list, y))
    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(y_true_list, 'r+', label='real data')
    plt.plot(y_pred_list, 'b*', label='pred data')
    plt.legend()
    plt.savefig('pre_temperature.jpg', dpi=256)
    plt.close()


if __name__ == '__main__':
    start_time = time.time()
    model = tm.TemperatureModel()
    train_loader = DataLoader(td.TemperatureDataset("meant.csv", "train"), batch_size=64, shuffle=True)
    test_loader = DataLoader(td.TemperatureDataset("meant.csv", "test"), batch_size=64, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)
    criterion = nn.MSELoss()
    epoch = 3000
    train(model, train_loader, criterion, optimizer, epoch)
    # 保存模型
    print("训练完成")
    torch.save(model, "temperature_model.pth")
    print("模型已保存")
    model = torch.load("temperature_model.pth")
    test(model, test_loader)
    print("总耗时:{}min".format((time.time() - start_time) / 60))
