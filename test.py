# -*- coding: utf-8 -*-
'''
测试各种小方法
'''
# @Time : 2021/3/18 17:56 
# @Author : LINYANZHEN
# @File : test.py
from matplotlib import pyplot
import numpy as np
import datetime
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# sigmod=1/w
def d_sigmoid_d_w(w):
    return -1 / pow(w, 2)


# w=1+b
def d_sigmoid_d_b(b):
    return d_sigmoid_d_w(1 + b) * 1


# b=e^a
def d_sigmoid_d_a(a):
    return d_sigmoid_d_b(np.exp(a)) * np.exp(a)


# a=-x
def d_sigmoid_d_x(x):
    return d_sigmoid_d_a(-x) * -1


# x_list = np.arange(-10, 10, 0.001)
# pyplot.plot(x_list, d_sigmoid(x_list), label="origin")
# pyplot.plot(x_list, d_sigmoid_d_x(x_list), label="new")
# pyplot.legend()
# pyplot.show()


# w = np.random.rand(4, 1)
# b = np.random.rand(4, 1)
# print(w.shape)
# print(b.shape)
# print((w + b).shape)

data = [1, 1, 1, 1, 1, 1]
w = [1, 2, -1, -4, 3, 1]
w = np.array(w)
tw = np.zeros((len(w) - 2))
# for i in range(30):
#     new = 0
#     for j in range(6):
#         new += data[-(j + 1)] * w[j]
#     data.append(new)
# df = pd.DataFrame(data)
# df.columns = ["meant"]
# df.to_csv("test.csv")
print(w[2:] - w[2 - 1:-1])
tw[:] = w[2:] - w[2 - 1:-1]
