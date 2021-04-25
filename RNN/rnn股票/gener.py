import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from net import Net
from readtxt import getdata

net=Net()
net.load_state_dict(torch.load(('netstate0.pkl')))
#data
mydict=getdata(1)
##make data
charnum=len(mydict)
mydata=np.random.randint(charnum,size=60)
mydata=list(mydata)

finalen=1000
for i in range(finalen):
    x00=torch.tensor(mydata[-60:],dtype=torch.float32)
    x00=x00.view(1,1,x00.size(0))
    out=net(x00)
    outlabel=torch.max(out,2)[1].data.numpy()[0][0]
    mydata.append(outlabel)
    

