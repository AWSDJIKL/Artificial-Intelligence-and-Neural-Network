# -*- coding: utf-8 -*-
'''
作业4
'''
# @Time : 2021/3/18 15:37 
# @Author : LINYANZHEN
# @File : hw4.py

import numpy as np
import random, matplotlib


class linear:
    def __init__(self, input_dimension, output_dimension):
        # 该层的权重
        # self.w = np.random.rand(output_dimension, input_dimension)
        self.w = np.ones((input_dimension, output_dimension), dtype=float) * 0.5
        # 该层的偏置量
        # self.b = np.random.rand(output_dimension, 1)
        self.b = np.ones((1, output_dimension), dtype=float) * 0.5
        # 记录该层输入进来的值
        self.input = np.random.rand(input_dimension, 1)
        # 记录该层输出的值
        self.output = np.random.rand(output_dimension, 1)

        # w和b的梯度
        self.gw = np.zeros((input_dimension, output_dimension), dtype=float)
        self.gb = np.zeros((1, output_dimension), dtype=float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 激活函数求导
    def d_sigmoid_dy(self, y, *args):
        return self.d_output_d_sigmoid(*args) * self.sigmoid(y) * (1 - self.sigmoid(y))

    # y=wx+b
    def d_sigmoid_dw(self, *args):
        y = np.matmul(self.input, self.w) + self.b
        return np.matmul(self.d_sigmoid_dy(y, *args), self.input)

    # y=wx+b
    def d_sigmoid_db(self, *args):
        y = np.matmul(self.input, self.w) + self.b
        return self.d_sigmoid_dy(y, *args) * 1

    def d_sigmoid_dx(self, *args):
        y = np.matmul(self.input, self.w) + self.b
        return np.matmul(self.d_sigmoid_dy(y, *args), self.w)

    # 前馈，计算输出值
    def forward(self):
        y = np.matmul(self.input, self.w) + self.b
        print("y", str(y))
        self.output = self.sigmoid(y)
        return

    # 反向传播，计算梯度
    def backward(self, fun, *args):
        # 链式求导中该层前面的求导结果
        self.d_output_d_sigmoid = fun
        w = self.d_sigmoid_dw(*args)
        b = self.d_sigmoid_db(*args)
        return w, b


x1 = np.load("x1_2.npy")
x2 = np.load("x2_2.npy")
y = np.load("y_2.npy")


def loss(y_pre, y):
    '''
    损失函数

    :param y_pre: 预测值
    :param y: 实际值
    :return: 均方差
    '''
    return pow((y_pre - y), 2) / 2


# 损失函数对y_pre求导
def d_loss_d_y_pre(*args):
    # return y_pre - y

    return args[0][0] - args[0][1]


# y = np.random.rand(4, 4)
# y_pre = np.random.rand(4, 4)

x = np.array([10, 10])
y = np.array([20])

l2 = linear(4, 1)
l1 = linear(2, 4)

l1.input = x
l1.forward()
l2.input = l1.output
l2.forward()

print("l1：" + str(l1.output))
print("l2:" + str(l2.output))
print("loss:", loss(l2.output, y))
print(l2.backward(d_loss_d_y_pre, (l2.output, y,)))
print(l1.backward(l2.d_sigmoid_dx, (l1.output,)))
