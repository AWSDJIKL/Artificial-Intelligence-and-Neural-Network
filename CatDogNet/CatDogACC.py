# -*- coding: utf-8 -*-
'''
测试模型在测试集上的准确率
'''
# @Time : 2021/4/9 22:47 
# @Author : LINYANZHEN
# @File : CatDogACC.py

import torch
import CatDogBuildDataset as CDBD
from torch.utils.data import DataLoader
import time
import numpy as np


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


if __name__ == '__main__':
    start_time = time.time()
    model_path = "model/CatDogTrain_32.pth"
    model = torch.load(model_path)
    train_set = CDBD.CDDataset(mode="train")
    test_set = CDBD.CDDataset(mode="test")
    train_loader = DataLoader(dataset=train_set, batch_size=35, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_set, batch_size=35, shuffle=True, num_workers=8)
    test(model, train_loader, test_loader)
    end_time = time.time()
    print("用时：{}min".format((end_time - start_time) / 60))
