import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import sys

class Conv1dGLU(nn.Module):
    '''
    Implementation of the Conv1d + GLU(Gated Linear Unit)
    with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    '''
    def __init__(self, in_channels=128, out_channels=128,padding = None,
                dilation = 2,kernel_size=12,*args, **kwargs):
        super(Conv1dGLU, self).__init__()
        if padding == None:
            padding = int(((kernel_size-1)/2)*dilation)
        self.conv1 = nn.Conv1d(in_channels, out_channels=out_channels,
                                padding=padding,dilation = dilation,
                                kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels, out_channels = out_channels,
                                padding=padding,dilation = dilation,
                                kernel_size=kernel_size)
    def forward(self, x):
        residual = x
        # print(x.size())
        x1 = self.conv1(x)
        # print(x1.shape)
        x2 = self.conv2(x)
        # print(x2.shape)
        # GLU
        x = x1 * torch.sigmoid(x2)
        x += residual
        x *= math.sqrt(0.5)
        return x
