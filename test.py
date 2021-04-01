# -*- coding: utf-8 -*-
'''
测试各种小方法
'''
# @Time : 2021/3/18 17:56 
# @Author : LINYANZHEN
# @File : test.py
from matplotlib import pyplot
import numpy as np


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


w = np.random.rand(4, 1)
b = np.random.rand(4, 1)
print(w.shape)
print(b.shape)
print((w + b).shape)
