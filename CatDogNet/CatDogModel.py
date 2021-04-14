# -*- coding: utf-8 -*-
'''
猫狗分类
'''
# @Time : 2021/4/8 17:14 
# @Author : LINYANZHEN
# @File : CatDogModel.py


import torch.nn as nn


class CDNet(nn.Module):
    def __init__(self):
        super(CDNet, self).__init__()
        self.convent = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),

            # 尝试修改模型以提升测试集准确率
            # nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.8),
            # nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.8),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            # nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        )
        self.linenet = nn.Sequential(
            nn.Linear(64 * 64 * 8, 1000),
            # nn.Linear(62 * 62 * 8, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.8),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.8),
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)

        )

    def forward(self, x):
        x = self.convent(x)
        # x = x.view(x.size(0), 64 * 64 * 8)
        # 将多维度tensor展开为1维
        x = x.view(x.size(0), -1)
        out = self.linenet(x)
        return out
