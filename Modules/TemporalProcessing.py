import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Modules.Conv1dGLU import Conv1dGLU

N_samples = 23

def Temp_Masking(x):
    '''
    Create function for temporal masking. Use librosa.decompose.hpss.
    Split and concatinate dimensions to make it 2D.

    '''
    pass


class TemporalProcessing(nn.Module):
    '''
    Implementation of Temporal Processing Layers
    '''

    def __init__(self,in_channels=128, out_channels=128,padding = None,
                dilation = 2,kernel_size=12):
        super(TemporalProcessing,self).__init__()
        self.conv1d_glu = Conv1dGLU(in_channels,out_channels,padding,dilation,
                                    kernel_size)



    def forward(self,x):
        batch_size = x.size(0)
        # transpose to do operation on the temporal dimension
        x = x.view(batch_size*N_samples, x.size(2), x.size(3)).transpose(1,2)
        x = self.conv1d_glu(x)
        x = x.transpose(1,2)

        x.contiguous()
        x = x.view(batch_size,N_samples,x.size(1),x.size(2))
        #x = librosa.decompose.hpss(x)[0]
        # temporal masking on x
        x = x.mean(dim=2)

        return x
