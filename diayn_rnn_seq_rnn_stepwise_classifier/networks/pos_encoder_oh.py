import torch
from torch import nn
import math

from code_slac.network.base import BaseNetwork

import self_supervised.utils.my_pytorch_util as my_ptu


class PositionalEncodingOh(BaseNetwork):

    def __init__(self,
                 ):
        super(PositionalEncodingOh, self).__init__()

    def forward(self, x):
        """
        Args:
            x           : (N, S, dim)
        Return:
            x           : (N, S, dim + S)
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = x.size(batch_dim)
        seq_len = x.size(seq_dim)
        x_dim = x.size(data_dim)
        oh = my_ptu.eye(seq_len)
        oh_rep = torch.stack([oh] * batch_size, dim=batch_dim)
        assert oh_rep.shape == torch.Size((batch_size, seq_len, seq_len))

        out = torch.cat([x, oh_rep], dim=data_dim)
        assert out.shape == torch.Size((batch_size, seq_len, x_dim + seq_len))

        return out





