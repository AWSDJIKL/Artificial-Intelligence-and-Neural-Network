import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from net import Net
from readtxt import getdata


#data
mydict,xtrain,ytrain=getdata(2)
xtrain=torch.tensor(xtrain,dtype=torch.float32)
ytrain=torch.tensor(ytrain,dtype=torch.long)
##net
net=Net()
##optimizer
opt=torch.optim.Adam(net.parameters(),lr=0.1)
##loss
loss_func=nn.CrossEntropyLoss()
##
mini_batch=640
seq_len=ytrain.shape[0]//mini_batch
totalnum=ytrain.shape[0]
#x00=xtrain[:mini_batch*seq_len,:].view(seq_len,mini_batch,xtrain.size(1))
#y00=ytrain[:mini_batch*seq_len].view(seq_len,mini_batch)
losslist=[]
for i in range(20):
  for j in range(seq_len):
    print(i)
    x00=xtrain[mini_batch*j:mini_batch*(j+1),...]
    x00=x00.view(x00.size(0),1,x00.size(1))
    y00=ytrain[mini_batch*j:mini_batch*(j+1)]
    out=net(x00)
    out=out.view(out.size(0),out.size(2))
    loss=loss_func(out,y00)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losslist.append(loss.item())
    print(loss.item())
torch.save(net.state_dict(),'netstate0.pkl')
##plot
losslist=np.array(losslist)
plt.plot(losslist)
plt.title('loss_lstm')
plt.savefig('loss_lstm_0.jpg',dpi=256)
plt.close()


