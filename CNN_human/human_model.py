# -*- coding: utf-8 -*-
'''
卷积神经网络分类人类动作模型
'''
# @Time : 2021/4/15 16:21 
# @Author : LINYANZHEN
# @File : human_model.py

import torch
import torch.nn as nn


class HumanModel(nn.Module):
    def __init__(self):
        super(HumanModel, self).__init__()

    def forward(self, x):
        return
