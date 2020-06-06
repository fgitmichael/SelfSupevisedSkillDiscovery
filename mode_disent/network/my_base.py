import torch
import torch.nn as nn

from code_slac.network.base import BaseNetwork, create_linear_network

class EncoderStateRep(BaseNetwork):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_units,
                 leaky_slope=0.2):
        super(EncoderStateRep, self).__init__()

        self.net = create_linear_network(
            input_dim,
            output_dim,
            hidden_units=hidden_units)

    def forward(self, x):
        return self.net(x)