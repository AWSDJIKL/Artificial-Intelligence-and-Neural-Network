# -*- coding: utf-8 -*-
'''
创建训练集
'''
# @Time : 2021/4/15 15:54 
# @Author : LINYANZHEN
# @File : human_build_dataset.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def normalized_data(data):
    '''
    数据归一化

    :param data:
    :return:
    '''
    data_max = np.max(data)
    data_min = np.min(data)
    return np.array([(i - data_min) / (data_max - data_min) for i in data])


def init_stock_data(train_set_size):
    '''
    数据初始化

    :param train_set_size: 训练集大小
    :return:
    '''
    # train_set_size = 50000
    # 读取数据
    data = pd.read_csv("pinganbank_5min.csv", delimiter=",")
    open_data = data["open"].values
    high_data = data["high"].values
    low_data = data["low"].values
    close_data = data["close"].values

    # 制造容器
    data_size = len(data)
    ts_size = 60
    open_x = np.zeros((data_size - ts_size, ts_size))
    high_x = np.zeros((data_size - ts_size, ts_size))
    low_x = np.zeros((data_size - ts_size, ts_size))
    close_x = np.zeros((data_size - ts_size, ts_size))
    label = np.zeros((data_size - ts_size))
    # print(len(label))
    # print(close_data[ts_size:])
    # print(close_data[ts_size - 1:-1])
    # print(len(close_data[ts_size:] - close_data[ts_size - 1:-1]))
    # print(close_data[ts_size:] - close_data[ts_size - 1:-1])
    # 填充数据
    for i in range(ts_size):
        open_x[:, i] = normalized_data(open_data[i:(data_size - ts_size + i)])
        high_x[:, i] = normalized_data(high_data[i:(data_size - ts_size + i)])
        low_x[:, i] = normalized_data(low_data[i:(data_size - ts_size + i)])
        close_x[:, i] = normalized_data(close_data[i:(data_size - ts_size + i)])
    label[:] = close_data[ts_size:] - close_data[ts_size - 1:-1]
    label = (label > 0) * 1

    # 放进x
    x = np.zeros((4, data_size - ts_size, ts_size))
    x[0, :, :] = open_x
    x[1, :, :] = high_x
    x[2, :, :] = low_x
    x[3, :, :] = close_x
    np.save("stock_train_data.npy", x[:, :train_set_size, :])
    np.save("stock_train_label.npy", label[:train_set_size])
    np.save("stock_test_data.npy", x[:, train_set_size:, :])
    np.save("stock_test_label.npy", label[train_set_size:])


class StockDataset(Dataset):
    def __init__(self, mode="train"):
        '''
        股票数据集类

        :param mode: 是训练/测试集
        '''
        super(StockDataset, self).__init__()
        if mode == "train":
            self.x = torch.tensor(np.load("stock_train_data.npy"), dtype=torch.float32)
            self.label = torch.tensor(np.load("stock_train_label.npy"), dtype=torch.long)
        elif mode == "test":
            self.x = torch.tensor(np.load("stock_test_data.npy"), dtype=torch.float32)
            self.label = torch.tensor(np.load("stock_test_label.npy"), dtype=torch.long)

    def __getitem__(self, index):
        return self.x[:, index, :], self.label[index]

    def __len__(self):
        return len(self.label)


if __name__ == '__main__':
    init_stock_data(50000)
    # train_set = StockDataset()
    # print(train_set[0])
