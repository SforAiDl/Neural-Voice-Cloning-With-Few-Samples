import torch
import torch.nn as nn
import torch.nn.functional as F


class PreNet(nn.Module):
    '''
    2-layer prenet
    1st is the linear layer.2nd is the elu activation layer
    '''

    def __init__(self , f_mel=80,f_mapped=128):
        super(PreNet,self).__init__()
        self.linear_1 = nn.Linear(f_mel,f_mapped)

    def forward(self,x):
        x = F.elu(self.linear_1(x))
        return x

class SpectralProcessing(nn.Module):
    '''
    Spectral Transformation layer that transforms mel
    spectogram to size 128
    '''

    def __init__(self,f_mel=80):
        super(SpectralProcessing,self).__init__()
        self.prenet_1  = PreNet(f_mel,128)

    def forward(self,x):
        mapped_x = self.prenet_1(x)

        return mapped_x
