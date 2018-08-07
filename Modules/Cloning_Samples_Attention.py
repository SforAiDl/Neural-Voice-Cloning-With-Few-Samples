
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.parameter as parameter

class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_units, dropout_p=0.5, h=2, is_masked=False):
        super(MultiHeadAttention, self).__init__()

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of query_dim and num_units must be the same")

        if torch.cuda.is_available:
            self.cuda=True

        self._num_units = num_units
        self._h = h
        if self.cuda:
            self._key_dim = Variable(torch.cuda.FloatTensor([key_dim]))
        else:
            self._key_dim = Variable(torch.FloatTensor([key_dim]))
        self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)
        self.bn = nn.BatchNorm1d(num_units)

    def forward(self, query, keys):
        Q = F.elu(self.query_layer(query))
        K = F.elu(self.key_layer(keys))
        V = F.elu(self.value_layer(keys))

        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        attention = torch.matmul(Q, K.transpose(1, 2))
        attention = attention / torch.sqrt(self._key_dim)

        if self._is_masked:
            diag_vals = attention[0].sign().abs()
            diag_mat = diag_vals.tril()
            diag_mat = diag_mat.unsqueeze(0).expand(attention.size())

            mask = Variable(
                torch.ones(diag_mat.size()).cuda.FloatTensor * (-2**32 + 1), requires_grad=False)

            attention = (attention * diag_mat) + (mask * (diag_mat-1).abs())
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self._dropout_p)
        attention = torch.matmul(attention, V)
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)
        attention += query
        attention = attention.transpose(1, 2)
        attention.contiguous()
        attention = self.bn(attention).transpose(1, 2)

        return attention
