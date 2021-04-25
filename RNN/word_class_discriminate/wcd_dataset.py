# -*- coding: utf-8 -*-
'''
词性判别——数据预处理
'''
# @Time : 2021/4/25 0:09 
# @Author : LINYANZHEN
# @File : wcd_dataset.py
import torch
import torch.nn as nn


def trans(seq, toindex):
    output = [toindex[i] for i in seq]
    return output


def get_indtag():
    # 训练数据，分词+对应词性
    train_data = [
        ("A simple lookup table that stores embeddings of a fixed dictionary and size".split(),
         []),
        ("This module is often used to store word embeddings and retrieve them using indices".split(),
         []),
        ("The input to the module is a list of indices, and the output is the corresponding word embeddings".split(),
         [])
    ]
    # 测试数据，仅分词
    test_data = [("This is a simple list of word").split()]
    # 为词赋予编号
    word_to_index = {}
    for sent, tags in train_data:
        for word in sent:
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
    # 为词赋予词性tag
    tag_to_index = {}
    train_x = []
    train_y = []
    for i in train_data:
        train_x.append(torch.tensor(trans(i[0], word_to_index), dtype=torch.long))
        train_y.append(torch.tensor(trans(i[1], tag_to_index), dtype=torch.long))
    test_x = trans(test_data[0], word_to_index)
    test_x = torch.tensor(test_x, dtype=torch.long)
    print(tag_to_index)
    print(word_to_index)
    return train_x, train_y, test_x
