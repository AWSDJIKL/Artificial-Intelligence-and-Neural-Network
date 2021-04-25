import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def getdata():
    input_word='abcde'
    w_to_id={'a':0,'b':1,'c':2,'d':3,'e':4}
    id_to_onehot={
            0:[1,0,0,0,0],
            1:[0,1,0,0,0],
            2:[0,0,1,0,0],
            3:[0,0,0,1,0],
            4:[0,0,0,0,1]}
    xtrain=[]
    for i in ['abcd','bcde','cdea','deab','eabc']:
        meow=[]
        for j in i:
            meow.append(id_to_onehot[w_to_id[j]])
        xtrain.append(meow)
    ytrain=[]
    for i in 'eabcd':
        ytrain.append(w_to_id[i])
    xtrain=np.array(xtrain)
    xtrain=torch.tensor(xtrain,dtype=torch.float32)
    ytrain=torch.tensor(ytrain,dtype=torch.long)
    return xtrain,ytrain

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.myrnn=nn.Sequential(
                nn.RNN(input_size=5,hidden_size=3,num_layers=1)
                )
        self.mylin=nn.Sequential(
                nn.Linear(3*4,5),
                nn.Softmax(dim=1)
                )
    def forward(self,x):
        output,hn=self.myrnn(x)
        output=output.view(output.size(0),3*4)
        output=self.mylin(output)
        return output

#data
xtrain,ytrain=getdata()
#net
net=Net()
#optimzer
opt=torch.optim.SGD(net.parameters(),lr=0.1)
#loss
loss_func=nn.CrossEntropyLoss()
#train
losslist=[]
for i in range(1000):
    out=net(xtrain)
    loss=loss_func(out,ytrain)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losslist.append(loss.item())
    print('%s,%4.3f'%(i,loss))
torch.save(net,'net2.pkl')
##plot
losslist=np.array(losslist)
plt.plot(losslist)
plt.title('loss2')
plt.savefig('loss2.jpg',dpi=256)
plt.close()
##predit
out=net(xtrain)
out=torch.max(out,1)[1].data.numpy()
print(out)
for i in range(xtrain.shape[0]):
    x0=xtrain[i,...]
    out0=out[i]
    y0=ytrain[i].numpy()
    print(x0,y0,out0)
    print('-----------')
