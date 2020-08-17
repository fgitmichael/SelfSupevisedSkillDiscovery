import torch
from torch import nn
from self_supervised.base.network.mlp import MyMlp

from code_slac.network.base import BaseNetwork

import rlkit.torch.pytorch_util as ptu


class NoohPosEncoder(BaseNetwork):

    def __init__(self,
                 encode_dim,
                 max_seq_len,):
        super(NoohPosEncoder, self).__init__()
        #self.encodings = torch.randn(
        #    max_seq_len,
        #    encode_dim,
        #)
        #self.encodings = torch.nn.Parameter(
        #    self.encodings,
        #    requires_grad=True).to(ptu.device)
        self.encoding_gen = MyMlp(
            input_size=encode_dim,
            output_size=encode_dim,
        )
        self.batch_dim = 0
        self.seq_dim = 1
        self.data_dim = -1

        self.seq_len = max_seq_len
        self.encode_dim = encode_dim

    @property
    def encodings(self):
        in_ = ptu.ones(self.seq_len, self.encode_dim)
        return self.encoding_gen(in_)

    def forward(self, seq):
        """
        Args:
            seq         : (N, S, dim)
        Return:
            seq_enc     : (N, S, dim + encode_dim)
        """
        batch_size = seq.size(self.batch_dim)
        seq_len = seq.size(self.seq_dim)

        if seq_len > self.seq_len:
            raise ValueError('Sequence is too long')

        encoding = torch.stack(
            [self.encodings[:seq_len]] * batch_size,
            dim=self.batch_dim
        )
        return torch.cat([seq, encoding], dim=self.data_dim)
