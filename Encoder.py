import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import librosa
import torch.nn.functional as F
from Modules.SpectralProcessing import SpectralProcessing
from Modules.TemporalProcessing import TemporalProcessing
from Modules.CloningSamplesAttention import CloningSamplesAttention


class Encoder(nn.Module):
    global batch_size
    global N_samples
    def __init__(self):
        super(Encoder, self).__init__()
        self.spectral_layer = SpectralProcessing(80)
        self.temporal_layer = TemporalProcessing()
        self.cloning_attention_layer = CloningSamplesAttention()

    def forward(self, x):
        #print(x)
        x = self.spectral_layer(x)
        x = self.temporal_layer(x)
        x = self.cloning_attention_layer(x)

        print(x.size())

        return x



#def Temp_Masking(x):
#Create function for temporal masking. Use librosa.decompose.hpss. Split and concatinate dimensions to make it 2D.
