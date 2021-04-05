import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def my_readcsv():
    data = pd.read_csv('data_x1x2x3y.csv', sep=',')
    train = data.head(80).values * 1.0
    test = data.tail(20).values * 1.0
    test[:, -1] = (test[:, -1] > 0) * 1.
    train[:, -1] = (train[:, -1] > 0) * 1.
    return train, test


def to_tensor(data):
    x = torch.tensor(data[:, :-1], dtype=torch.float32)
    y = torch.tensor(data[:, -1].reshape(-1, 1), dtype=torch.float32)
    return x, y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.predict = nn.Sequential(
            nn.Linear(3, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        prediction = self.predict(x)
        return prediction


# get test and train
train, test = my_readcsv()
xtrain, ytrain = to_tensor(train)
xtest, ytest = to_tensor(test)
##get net
net = Net()
##optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
##loss
loss_func = nn.MSELoss()
##train
for i in range(200):
    out = net(xtrain)
    loss = loss_func(out, ytrain)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # draw
    if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 30, 50, 70, 100, 150, 199]:
        print(i)
        plt.figure(figsize=(16, 9))
        plt.plot(out.data.numpy(), 'ro', label='train')
        plt.plot(ytrain.data.numpy(), 'k*', label='real')
        plt.title('train, %d, loss=%f' % (i, loss))
        plt.legend()
        plt.savefig('p0_train_%d.jpg' % i, dpi=256)
        plt.close()

##test
pred = net(xtest)
predloss = loss_func(pred, ytest)
plt.figure(figsize=(16, 9))
plt.plot(pred.data.numpy(), 'ro', label='test')
plt.plot(ytest.data.numpy(), 'k*', label='real')
plt.title('test loss=%f' % (predloss))
plt.legend()
plt.savefig('p0_test.jpg', dpi=256)
plt.close()
