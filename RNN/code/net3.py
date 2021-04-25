import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.myrnn=nn.Sequential(
                nn.RNN(input_size=1,hidden_size=20,num_layers=1,dropout=0.2)
                )
        self.mylin=nn.Sequential(
                nn.Linear(20*60,1)
                )
    def forward(self,x):
        output,hn=self.myrnn(x)
        output=output.view(output.size(0),20*60)
        output=self.mylin(output)
        return output

