import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from Modules.Attention import Attention

class CloningSamplesAttention(nn.Module):
    '''
    Implementation of the the last Cloning sample attention part.
    Implementation includes residual linear connection,Multiheadattentionlayer,
    and linear layers.
    '''

    def __init__(self):
        super(CloningSamplesAttention,self).__init__()
        self.residual_linear_layer = nn.Linear(128,512)
        self.attention = Attention(128)
        self.fc_after_attention = nn.Linear(128,1)

    def forward(self,x):

        residual_linear_x = self.residual_linear_layer(x)
        x.contiguous()
        # attention layer
        x = self.attention(x)
        # linear layers
        x = self.fc_after_attention(x)
        x = torch.squeeze(x)
        x = F.softsign(x)
        x = F.normalize(x, dim = 1)
        x = torch.unsqueeze(x, dim=2)
        x = torch.bmm(x.transpose(1,2), residual_linear_x)
        x = torch.squeeze(x)

        return x
