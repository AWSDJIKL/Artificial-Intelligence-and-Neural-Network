# -*- coding: utf-8 -*-
'''
训练模型并保存
'''
# @Time : 2021/4/15 17:36 
# @Author : LINYANZHEN
# @File : stock_train.py

import torch
import stock_model as sm
import stock_build_dataset as sbd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time


def test(model, train_loader, test_loader):
    train_set_acc = 0
    num = 0
    for i, (x, y) in enumerate(train_loader, 0):
        out = model(x)
        # detach()表示去掉梯度信息
        out = out.detach().numpy()
        y = y.numpy()
        # 找出每行最大值的下表(即预测结果中概率最大的项)
        out = np.argmax(out, axis=1)
        # 与真实值对比
        result = out == y
        train_set_acc += sum(result == True)
        num += len(result)
    train_set_acc /= num

    test_set_acc = 0
    num = 0
    for i, (x, y) in enumerate(test_loader, 0):
        out = model(x)
        # detach()表示去掉梯度信息
        out = out.detach().numpy()
        y = y.numpy()
        # 找出每行最大值的下表(即预测结果中概率最大的项)
        out = np.argmax(out, axis=1)
        # 与真实值对比
        result = out == y
        print("out", out)
        print("y", y)
        # print(result)
        test_set_acc += sum(result == True)
        num += len(result)
    test_set_acc /= num
    print("  训练集正确率：{}".format(train_set_acc))
    print("  测试集正确率：{}".format(test_set_acc))
    return train_set_acc, test_set_acc


def train(model, train_loader, test_loader, criterion, optimizer, epoch):
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    for e in range(epoch):
        print("当前训练轮次：({}/{})".format(e, epoch))
        epoch_start_time = time.time()
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
        train_acc, test_acc = test(model, train_loader, test_loader)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        epoch_end_time = time.time()
        print("总耗时：{}min".format((epoch_end_time - epoch_start_time) / 60))
    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, "b")
    plt.title('loss')
    plt.savefig('Stock_Train_loss.jpg', dpi=256)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_list, "b", label='train acc')
    plt.plot(test_acc_list, "r", label='test acc')
    plt.legend()
    plt.grid()
    plt.title('acc')
    plt.savefig('Stock_Train_acc.jpg', dpi=256)
    plt.close()

    return


if __name__ == '__main__':
    start_time = time.time()
    train_loader = DataLoader(sbd.StockDataset(mode="train"), batch_size=40, shuffle=True)
    test_loader = DataLoader(sbd.StockDataset(mode="test"), batch_size=40, shuffle=False)
    model = sm.StockModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    epoch = 1
    train(model, train_loader, test_loader, criterion, optimizer, epoch)
    torch.save(model, "Stock_Train.pth")
    end_time = time.time()
    print("总耗时：{}min".format((end_time - start_time) / 60))
