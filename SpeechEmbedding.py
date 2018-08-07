
# coding: utf-8

# In[2]:


import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import librosa
import torch.nn.functional as F
from Modules.SpectralProcessing import PreNet
from  Modules.Conv1dGLU import Conv1dGLU
from Modules.Encoder import Attention


# In[ ]:


batch_size = 64
N_samples = 23


# In[55]:


class Encoder(nn.Module):
    global batch_size
    global N_samples
    def __init__(self):
        super(Encoder, self).__init__()
        self.prenet = PreNet()
        self.conv = Conv1dGLU()
        self.attention = Attention(128)
        self.prohead = nn.Linear(128,1)
        self.residual_conv = nn.Linear(128,512)
        self.bn = nn.BatchNorm1d(N_samples)

    def forward(self, x):
        #print(x)
        x = self.prenet(x)
        x = x.view(batch_size*N_samples, x.size(2), x.size(3)).transpose(1,2)
        x = self.conv(x)
        x = x.transpose(1,2)
        x.contiguous()
        x = x.view(batch_size,N_samples,x.size(1),x.size(2))
        #x = librosa.decompose.hpss(x)[0]
        x = x.mean(dim=2)
        conv_out = x
        conv_out = self.residual_conv(conv_out)
        x.contiguous()
        #print(x)
        x = self.attention(x)
        #print(x)
        x = self.prohead(x)
        x = torch.squeeze(x)
        x = F.softsign(x)
        x = self.bn(x)
        x = torch.unsqueeze(x, dim=2)
        x = torch.bmm(x.transpose(1,2), conv_out)
        x = torch.squeeze(x)
        return x


# In[56]:


# enc = Encoder()


# In[57]:


# z = torch.randn(25,20,100,80)


# In[58]:


# out = enc(Variable(z))
# out


# In[63]:


#def Temp_Masking(x):
#Create function for temporal masking. Use librosa.decompose.hpss. Split and concatinate dimensions to make it 2D.
