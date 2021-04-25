# -*- coding: utf-8 -*-
'''
测试原版数据导入方式训练模型
'''
# @Time : 2021/4/23 21:31 
# @Author : LINYANZHEN
# @File : dataset_test.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import pandas as pd
import temperature_model as tm


def get_data(input_size):
    pd.set_option('display.max_rows', None)
    data = pd.read_csv("meant.csv", usecols=(1, 2), dtype={"date": str})
    # 数据预处理，去掉非法的日期
    # 去掉非润年的2月29日
    error_date = data.apply(lambda row: int(row["date"][:4]) / 4 != 0 and "0229" in row["date"], axis=1)
    # print(data[error_date])
    data = data.drop(data[error_date].index).reset_index(drop=True)
    # 去掉2月30日
    error_date = data.apply(lambda row: "0230" in row["date"], axis=1)
    # print(data[error_date])
    data = data.drop(data[error_date].index).reset_index(drop=True)
    # 去掉2，4，6，9，11月的31日
    d = ["0231", "0431", "0631", "0931", "1131"]
    error_date = data.apply(lambda row: any(x in row["date"] for x in d), axis=1)
    # print(data[error_date])
    data = data.drop(data[error_date].index).reset_index(drop=True)
    # 对数据按时间排序
    data['newdate'] = pd.to_datetime(data.date, format='%Y%m%d')
    # print(data)
    data = data.sort_values(by="newdate").reset_index(drop=True)
    # print(data)
    i = data[data["date"] == "19920101"].index[0]

    x_train = np.zeros((i - input_size, input_size))
    y_train = np.zeros((i - input_size, 1))
    for j in range(input_size):
        x_train[:, j] = data["meant"].iloc[j:i - (input_size - j)]
    y_train[:, 0] = data["meant"].iloc[input_size:i]
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))

    x_test = np.zeros((data.shape[0] - i - input_size, input_size))
    y_test = np.zeros((data.shape[0] - i - input_size, 1))
    for j in range(input_size):
        x_test[:, j] = data["meant"].iloc[i + j: -(input_size - j)]
    x_test[:, 0] = data["meant"].iloc[i + input_size:]
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    input_size = 60
    x_train, y_train, x_test, y_test = get_data(input_size)
    model = tm.TemperatureModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    epoch = 400
    batch_size = 64
    total_num = y_train.shape[0]
    loss_list = []
    for i in range(epoch):
        epoch_start_time = time.time()
        print("epoch[{}/{}]".format(i, epoch))
        for j in range(int(total_num / batch_size)):
            x00 = x_train[j * batch_size:(j + 1) * batch_size, ...]
            y00 = y_train[j * batch_size:(j + 1) * batch_size]
            out = model(x00)
            loss = criterion(out, y00)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print("  用时:{}min".format((time.time() - epoch_start_time) / 60))
    print("训练完成")
    torch.save(model, "pre_temperature_test.pth")
    print("模型已保存")
