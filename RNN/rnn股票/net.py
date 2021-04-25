import numpy as np
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.myrnn=nn.Sequential(
                nn.LSTM(input_size=3540,hidden_size=20,num_layers=1,dropout=0.3)
                )
        self.mylin=nn.Sequential(
                nn.Linear(20,59),
                nn.Softmax(dim=1)
                )
    def forward(self,x):
        output,hn=self.myrnn(x)
        #output=output.view(output.size(0),output.size(1),20)
        output=self.mylin(output)
        return output

