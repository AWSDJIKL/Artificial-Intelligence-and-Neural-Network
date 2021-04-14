# -*- coding: utf-8 -*-
'''
训练模型并保存
'''
# @Time : 2021/4/9 12:53 
# @Author : LINYANZHEN
# @File : CatDogTrain.py

import torch
import torch.nn as nn
import CatDogModel as CDM
import CatDogBuildDataset as CDBD
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import os
import CatDogACC as CDACC


def train(model, train_loader, test_loader, criterion, optimizer, epoch):
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    for e in range(epoch):
        print("当前训练轮次：{}".format(e))
        for i, (x, y) in enumerate(train_loader, 0):
            # 计算预测值
            out = model(x)
            # 梯度归零
            optimizer.zero_grad()
            # 计算损失，反向传播
            loss = criterion(out, y)
            loss.backward()
            # 更新权重
            optimizer.step()
            loss_list.append(loss.item())
        train_acc, test_acc = CDACC.test(model, train_loader, test_loader)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, "b")
    plt.title('loss')
    plt.savefig('CatDogTrain_loss.jpg', dpi=256)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_list, "b", label='train acc')
    plt.plot(test_acc_list, "r", label='test acc')
    plt.legend()
    plt.grid()
    plt.title('acc')
    plt.savefig('CatDogTrain_acc.jpg', dpi=256)
    plt.close()

    return


if __name__ == '__main__':
    start_time = time.time()
    model = CDM.CDNet()
    train_set = CDBD.CDDataset(mode="train")
    test_set = CDBD.CDDataset(mode="test")
    train_loader = DataLoader(dataset=train_set, batch_size=35, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_set, batch_size=30, shuffle=False, num_workers=8)
    # 分类问题常用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 使用随机梯度下降法作为优化器
    optimizer = torch.optim.ASGD(model.parameters(), lr=0.03)
    train(model, train_loader, test_loader, criterion, optimizer, epoch=1000)
    # 设置模型保存路径
    model_save_path = "model"
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    torch.save(model, os.path.join(model_save_path, "CatDogTrain.pth"))
    end_time = time.time()
    print("用时：{}min".format((end_time - start_time) / 60))
