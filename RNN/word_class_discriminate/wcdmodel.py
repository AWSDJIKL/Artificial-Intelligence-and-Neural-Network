# -*- coding: utf-8 -*-
'''
词性判别——模型
'''
# @Time : 2021/4/24 22:43 
# @Author : LINYANZHEN
# @File : wcd_model.py
import torch
import torch.nn as nn


class WCDModel(nn.Module):
    def __init__(self):
        super(WCDModel, self).__init__()
        # 分词，得到输入语句里的词的分布矩阵[语句中词的数量。词的种类]
        # 要求输入(词汇量大小，每个词汇向量表示的向量为度)
        self.word = nn.Embedding(9, 10)
        self.lstm = nn.LSTM(input_size=10, hidden_size=3, num_layers=1, dropout=0)
        self.linear = nn.Sequential(
            nn.Linear(3, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.word(x)
        x = x.unsqueeze(1)
        x, hn = self.lstm(x)
        x = x.view(x.size(0), x.size(2))
        x = self.linear(x)
        return x
