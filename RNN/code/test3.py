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

net=Net()
net.load_state_dict(torch.load('netstate3.pkl'))
xtrain,ytrain,xtest,ytest=getdata()

##predit
out=net(xtest)
print(out.shape,ytest.shape)
plt.plot(out.data.numpy()[:,0],color='b',label='rnn')
plt.plot(ytest.data.numpy(),color='r',label='real')
plt.legend()
plt.title('rnn vs real')
plt.savefig('test3.jpg',dpi=256)
plt.close()
