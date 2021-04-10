# -*- coding: utf-8 -*-
'''
数据集加载
'''
# @Time : 2021/4/8 22:58 
# @Author : LINYANZHEN
# @File : CatDogBuildDataset.py

import torch
from torch.utils.data import Dataset
import cv2
import random
import os
import numpy as np


def cat_dog_build_dataset(dataset_path):
    l = [i for i in range(100)]
    random.shuffle(l)
    train_set_data = np.zeros((140, 3, 128, 128))
    test_set_data = np.zeros((60, 3, 128, 128))
    train_set_label = np.zeros((140))
    test_set_label = np.zeros((60))
    # 取70张猫作为训练集
    for i in range(70):
        image = cv2.imread(os.path.join(dataset_path, "cat.{}.jpg".format(l[i])))
        image = cv2.resize(image, (128, 128))
        r = image[:, :, 2]
        g = image[:, :, 1]
        b = image[:, :, 0]
        train_set_label[i] = 0
        train_set_data[i, 0, :, :] = r
        train_set_data[i, 1, :, :] = g
        train_set_data[i, 2, :, :] = b
    # 剩下30张猫作为测试集
    for i in range(70, 100):
        image = cv2.imread(os.path.join(dataset_path, "cat.{}.jpg".format(l[i])))
        image = cv2.resize(image, (128, 128))
        r = image[:, :, 2]
        g = image[:, :, 1]
        b = image[:, :, 0]
        test_set_label[i - 70] = 0
        test_set_data[i - 70, 0, :, :] = r
        test_set_data[i - 70, 1, :, :] = g
        test_set_data[i - 70, 2, :, :] = b
    # 再随机取70张狗
    for i in range(70):
        image = cv2.imread(os.path.join(dataset_path, "dog.{}.jpg".format(l[i])))
        image = cv2.resize(image, (128, 128))
        r = image[:, :, 2]
        g = image[:, :, 1]
        b = image[:, :, 0]
        train_set_label[i + 70] = 1
        train_set_data[i + 70, 0, :, :] = r
        train_set_data[i + 70, 1, :, :] = g
        train_set_data[i + 70, 2, :, :] = b
    # 剩下30张狗作为测试集
    for i in range(70, 100):
        image = cv2.imread(os.path.join(dataset_path, "cat.{}.jpg".format(l[i])))
        image = cv2.resize(image, (128, 128))
        r = image[:, :, 2]
        g = image[:, :, 1]
        b = image[:, :, 0]
        test_set_label[i - 70 + 30] = 1
        test_set_data[i - 70 + 30, 0, :, :] = r
        test_set_data[i - 70 + 30, 1, :, :] = g
        test_set_data[i - 70 + 30, 2, :, :] = b
    print("train_set_data", train_set_data)
    print("test_set_data", test_set_data)
    print("train_set_label", train_set_label)
    print("test_set_label", test_set_label)
    np.save("train_set_data", train_set_data)
    np.save("test_set_data", test_set_data)
    np.save("train_set_label", train_set_label)
    np.save("test_set_label", test_set_label)


class CDDataset(Dataset):
    def __init__(self, mode="train"):
        super(CDDataset, self).__init__()
        self.lables = []
        self.images = []
        self.len = 0
        if mode == "train":
            # 是训练集
            self.lables = torch.tensor(np.load("train_set_label.npy"), dtype=torch.long)
            self.images = torch.tensor(np.load("train_set_data.npy") / 256.0, dtype=torch.float32)
            self.len = len(self.lables)
        elif mode == "test":
            self.lables = torch.tensor(np.load("test_set_label.npy"), dtype=torch.long)
            self.images = torch.tensor(np.load("test_set_data.npy") / 256.0, dtype=torch.float32)
            self.len = len(self.lables)

    def __getitem__(self, index):
        return self.images[index, :, :, :], self.lables[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    cat_dog_build_dataset("sample")
