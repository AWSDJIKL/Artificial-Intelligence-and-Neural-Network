# -*- coding: utf-8 -*-
'''
温度数据集
'''
# @Time : 2021/4/22 16:32 
# @Author : LINYANZHEN
# @File : temperature_dataset.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import pandas as pd


class TemperatureDataset(Dataset):
    def __init__(self, file_path, mode="train"):
        '''
        温度数据集

        :param file_path: 数据文件路径
        :param mode: 模式（训练集/测试集）
        '''
        super(TemperatureDataset, self).__init__()
        data = pd.read_csv(file_path, usecols=(1, 2), dtype={"date": str})
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
        # 归一化
        meant_max = data["meant"].max()
        meant_min = data["meant"].min()
        data["meant"] = (data["meant"] - meant_min) / (meant_max - meant_min)

        print(data)
        i = data[data["date"] == "19920101"].index[0]
        if mode == "train":
            x = np.zeros((i - 60, 60))
            y = np.zeros((i - 60, 1))
            for j in range(60):
                x[:, j] = data["meant"].iloc[j:i - (60 - j)]
            y[:, 0] = data["meant"].iloc[60:i]
            x = x.reshape((x.shape[0], 1, x.shape[1]))
        else:
            x = np.zeros((data.shape[0] - i - 60, 60))
            y = np.zeros((data.shape[0] - i - 60, 1))
            for j in range(60):
                x[:, j] = data["meant"].iloc[i + j: -(60 - j)]
            y[:, 0] = data["meant"].iloc[i + 60:]
            x = x.reshape((x.shape[0], 1, x.shape[1]))
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index].to(torch.float32), self.y[index].to(torch.float32)

    def __len__(self):
        return self.len
