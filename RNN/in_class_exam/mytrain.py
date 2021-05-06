# -*- coding: utf-8 -*-
'''

'''
# @Time : 2021/5/6 16:09 
# @Author : LINYANZHEN
# @File : mytrain.py
import pickle
import string
import torch
from myrnn1 import RNN
import makedata
import somefun
import numpy as np
import matplotlib.pyplot as plt
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

f = open('word_dict.pkl', 'rb')
category_lines = pickle.load(f)
f.close()
all_categories = list(category_lines.keys())
n_categories = len(all_categories)
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
criterion = torch.nn.NLLLoss()
opt = torch.optim.SGD(rnn.parameters(), lr=1e-2)


def train(category_tensor, line_tensor):
    opt.zero_grad()
    hidden = rnn.initHidden()
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()
    opt.step()
    return output, loss.item()


n_iters = 10000
print_every = 50
all_losses = []

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = somefun.randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    all_losses.append(loss)
    if iter % print_every == 0:
        guess, guess_i = somefun.categoryFromOutput(output)
        correct = "yes" if guess == category else "no (%s)" % category
        print("%04d %03d%%,loss=%.4f %s\t/ guess=%s correct=%s" % (
        iter, iter / n_iters * 100, loss, line, guess, correct))
torch.save(rnn.state_dict(),"net_myrnn1.pkl")
losslist=np.array(all_losses)
plt.plot(losslist)
plt.title("loss_rnn1")
plt.savefig("loss_rnn1.png",dpi=256)
plt.close()
