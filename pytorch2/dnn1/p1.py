import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def read_xydata():
    x=np.linspace(-np.pi,np.pi,100)
    y=np.sin(x)+0.5*np.random.rand(len(x))
    x=torch.tensor(x.reshape(-1,1),dtype=torch.float32)
    y=torch.tensor(y.reshape(-1,1),dtype=torch.float32)
    #x=torch.unsqueeze(torch.linspace(-np.pi,np.pi,100),dim=1)
    #y=torch.sin(x)+0.5*torch.rand(x.size())
    return x,y

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.predict=nn.Sequential(
                nn.Linear(1,10),
                nn.ReLU(),
                nn.Linear(10,1)
                )
    def forward(self,x):
        prediction=self.predict(x)
        return prediction


##readdata
x,y=read_xydata()
##get net
net=Net()
##optimizer
optimizer=torch.optim.SGD(net.parameters(),lr=0.05)
##loss
loss_func=nn.MSELoss()
##train
for i in range(10000):
    out=net(x)
    loss=loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ##draw
    if i in [0,1,2,10,25,50,100,200,300,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,9999]:
        print(i)
        plt.scatter(x,y,label='real')
        plt.plot(x,out.data.numpy(),'r',lw=5,label='train')
        plt.title('%d, loss=%f'%(i,loss))
        plt.savefig('p1_NET1_train_%d'%i)
        plt.close()

#torch.save(net,'p1_net.pkl')
#net=torch.load('p1_net.pkl')




