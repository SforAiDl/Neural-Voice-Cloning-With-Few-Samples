import torch
import torch.nn as nn
import torch.nn.functional as F


class PreNet(nn.Module):
    def __init__(self):
        super(PreNet,self).__init__()
        self.layer = nn.Linear(80,128)

    def forward(self,x):
        x = F.elu(self.layer(x))
        return x
