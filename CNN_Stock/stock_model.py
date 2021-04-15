# -*- coding: utf-8 -*-
'''
卷积神经网络预测股票走势模型
'''
# @Time : 2021/4/15 15:52 
# @Author : LINYANZHEN
# @File : human_model.py
import torch
import torch.nn as nn


class StockModel(nn.Module):
    def __init__(self):
        super(StockModel, self).__init__()
        self.relu = torch.nn.ReLU
        self.convent = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
        )
        self.linenet = nn.Sequential(
            nn.Linear(8 * 128, 1000, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.convent(x)
        # 将多维度tensor展开为1维
        x = x.view(x.size(0), -1)
        out = self.linenet(x)
        return out
