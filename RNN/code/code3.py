import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from net3 import Net

def getdata():
    myd=pd.read_csv('data\\pinganbank_day.csv',sep=',')
    myopen=myd.open.values
    omax=myopen.max()
    omin=myopen.min()
    myopen=(myopen-omin)/(omax-omin)
    xopen=np.zeros((myopen.shape[0]-60,60))
    for i in range(60):
        xopen[:,i]=myopen[i:myopen.shape[0]-60+i]
    yopen=myopen[60:]
    xopen=xopen.reshape(xopen.shape[0],xopen.shape[1],1)
    xopen=torch.tensor(xopen,dtype=torch.float32)
    yopen=torch.tensor(yopen,dtype=torch.float32)
    xtrain=xopen[:1100,:,:]
    ytrain=yopen[:1100]
    xtest=xopen[1100:,:,:]
    ytest=yopen[1100:]
    return xtrain,ytrain,xtest,ytest 


#data
xtrain,ytrain,xtest,ytest=getdata()
#net
net=Net()
#optimzer
opt=torch.optim.Adam(net.parameters(),lr=0.005)
#loss
loss_func=nn.MSELoss()
#train
mini_batch=64
totalnum=ytrain.shape[0]
losslist=[]
for i in range(50):
  for j in range(int(totalnum/mini_batch)):
    x00=xtrain[j*mini_batch:(j+1)*mini_batch,...]
    y00=ytrain[j*mini_batch:(j+1)*mini_batch]
    out=net(x00)
    loss=loss_func(out,y00)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losslist.append(loss.item())
    print('%s,%4.3f'%(i,loss))
#torch.save(net,'net3.pkl')
torch.save(net.state_dict(),'netstate3.pkl')
##plot
losslist=np.array(losslist)
plt.plot(losslist)
plt.title('loss3')
plt.savefig('loss3.jpg',dpi=256)
plt.close()

