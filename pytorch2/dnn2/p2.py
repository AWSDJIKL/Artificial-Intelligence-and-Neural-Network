import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    data=torch.ones(200,2)
    x0=torch.normal(2*data,1)
    y0=torch.zeros(200)
    x1=torch.normal(-2*data,1)
    y1=torch.ones(200)
    x=torch.cat((x0,x1)).type(torch.FloatTensor)
    y=torch.cat((y0,y1)).type(torch.LongTensor)
    return x,y

def get_testdata():
    data=torch.ones(50,2)
    x0=torch.normal(2*data,1)
    y0=torch.zeros(50)
    x1=torch.normal(-2*data,1)
    y1=torch.ones(50)
    x=torch.cat((x0,x1)).type(torch.FloatTensor)
    y=torch.cat((y0,y1)).type(torch.LongTensor)
    return x,y



class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.classify=nn.Sequential(
                nn.Linear(2,15),
                nn.ReLU(),
                nn.Linear(15,2),
                nn.Softmax(dim=1)
                )
    def forward(self,x):
        classification=self.classify(x)
        return classification

#readata
x,y=get_data()
#build net
net=Net()
##optimizer
optimizer=torch.optim.SGD(net.parameters(),lr=0.03)
loss_func=nn.CrossEntropyLoss()
for i in range(100):
    out=net(x)
    loss=loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #draw
    if i in [0,1,2,3,4,5,10,20,30,40,50,70,99]:
        print(i)
        myclass=torch.max(out,1)[1]
        myclass=myclass.data.numpy()
        myreal=y.data.numpy()
        x1=x.data.numpy()[:,0]
        x2=x.data.numpy()[:,1]
        plt.plot(x1[myreal==0],x2[myreal==0],'o',color='pink',label='real_0')
        plt.plot(x1[myreal==1],x2[myreal==1],'o',color='lightblue',label='real_1')
        plt.plot(x1[myclass==0],x2[myclass==0],'r*',label='train_0')
        plt.plot(x1[myclass==1],x2[myclass==1],'k*',label='train_1')
        plt.legend()
        plt.title('%d,loss=%f'%(i,loss))
        plt.savefig('p2_train_%d.jpg'%i,dpi=256)
        plt.close()

xtest,ytest=get_testdata()
out=net(xtest)
loss=loss_func(out,ytest)
myclass=torch.max(out,1)[1]
myclass=myclass.data.numpy()
myreal=ytest.data.numpy()
x1=xtest.data.numpy()[:,0]
x2=xtest.data.numpy()[:,1]
plt.plot(x1[myreal==0],x2[myreal==0],'o',color='pink',label='real_0')
plt.plot(x1[myreal==1],x2[myreal==1],'o',color='lightblue',label='real_1')
plt.plot(x1[myclass==0],x2[myclass==0],'r*',label='test_0')
plt.plot(x1[myclass==1],x2[myclass==1],'k*',label='test_1')
plt.legend()
plt.title('test,loss=%f'%(loss))
plt.savefig('p2_test.jpg',dpi=256)
plt.close()



