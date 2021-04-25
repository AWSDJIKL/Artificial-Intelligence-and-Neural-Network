# -*- coding: utf-8 -*-
'''
词性判别——训练
'''
# @Time : 2021/4/25 0:09 
# @Author : LINYANZHEN
# @File : wcd_train.py
import wcdmodel as wm
import wcd_dataset as wd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time


def train(model, criterion, optimizer, epoch):
    train_x, train_y, test_x = wd.get_indtag()
    loss_list = []
    for i in range(epoch):
        epoch_start_time = time.time()
        print("epoch[{}/{}]".format(i, epoch))
        for x, y in zip(train_x, train_y):
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print("  用时:{}min".format((time.time() - epoch_start_time) / 60))

    # 画出损失图像
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, "b")
    plt.title('loss')
    plt.savefig('wcd_Train_loss.jpg', dpi=256)
    plt.close()


def test(model):
    tag_to_index = {}
    index_to_tag = {}
    for i, j in tag_to_index.items():
        index_to_tag[j] = i
    train_x, train_y, test_x = wd.get_indtag()
    out = model(test_x)
    out_label = torch.max(out, 1)[1].data.numpy()
    print("test data:")
    print(wd.trans(out_label, index_to_tag))


if __name__ == '__main__':
    start_time = time.time()
    model = wm.WCDModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()

    epoch = 400
    train(model, criterion, optimizer, epoch)
    # 保存模型
    print("训练完成")
    torch.save(model, "wcd_model.pth")
    print("模型已保存")
    model = torch.load("wcd_model.pth")
    test(model)
    print("总耗时:{}min".format((time.time() - start_time) / 60))
