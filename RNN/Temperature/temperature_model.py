# -*- coding: utf-8 -*-
'''
循环神经网络预测温度模型
'''
# @Time : 2021/4/22 16:30 
# @Author : LINYANZHEN
# @File : temperature_model.py
import torch
import torch.nn as nn


class TemperatureModel(nn.Module):
    def __init__(self):
        super(TemperatureModel, self).__init__()
        self.rnn = nn.RNN(input_size=60, hidden_size=20, num_layers=3, dropout=0.5, batch_first=False)
        self.linear = nn.Linear(20, 1)

        # self.rnn = nn.Sequential(
        #     nn.RNN(input_size=60, hidden_size=20, num_layers=1, dropout=0, batch_first=False)
        # )
        # self.linear = nn.Sequential(
        #     nn.Linear(20, 1)
        # )

    def forward(self, x):
        output = self.rnn(x)
        output = self.linear(output[0])
        return output
