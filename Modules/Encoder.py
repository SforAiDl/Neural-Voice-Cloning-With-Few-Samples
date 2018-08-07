import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from .Cloning_Samples_Attention import MultiHeadAttention

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()

        self.encoders = self._build_model(dim)

    def _build_model(self, dim):
        layers = []
        dim = dim
        layers.append(MultiHeadAttention(dim, dim, dim))

        return nn.ModuleList(layers)

    def forward(self, inputs):
        net_inputs = inputs
        net_inputs.contiguous()
        for enc in self.encoders:
            net_inputs = enc(net_inputs, net_inputs)
        return net_inputs
