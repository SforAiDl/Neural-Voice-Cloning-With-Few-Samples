import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Conv1dGLU(nn.Module):
    '''Implementation of the conv1d + GLU'''
    def __init__(self, in_channels=128, out_channels=2*128, padding = None, dilation = 1,
                 kernel_size=12,*args, **kwargs):
        super(Conv1dGLU, self).__init__()
        if padding == None:
            padding = (kernel_size-1)//2*dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels=2*in_channels, padding=padding,
                               dilation = dilation, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels, out_channels = 2*in_channels, padding=padding+1,
                               dilation = dilation, kernel_size=kernel_size)
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = x.split(x.size(1)//2, dim = 1)#WHAT IS A GLU??
        x = x1 * F.sigmoid(x2)
        x = self.conv2(x)
        x1, x2 = x.split(x.size(1)//2, dim = 1)
        x = x1 * F.sigmoid(x2)
        x += residual
        x *= math.sqrt(0.5)
        return x
