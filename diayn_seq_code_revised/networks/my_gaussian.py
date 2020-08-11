import torch
import torch.nn as nn
from torch.distributions import Normal

from code_slac.network.latent import Gaussian, ConstantGaussian
from self_supervised.base.network.mlp import MyMlp


class MyGaussian(Gaussian):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_units,
                 std=None,
                 leaky_slope=0.2):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=hidden_units,
            std=std,
            leaky_slope=leaky_slope
        )

        # Overwrite net
        self.net = MyMlp(
            input_size=input_dim,
            output_size=2*output_dim if std is None else output_dim,
            hidden_sizes=hidden_units,
            hidden_activation=nn.LeakyReLU(leaky_slope)
        )

    @property
    def output_size(self):
        if self.std is None:
            assert self.net.output_size % 2 == 0

        _output_size = int(self.net.output_size/2) \
            if self.std is None \
            else self.net.output_size
        return _output_size


class ConstantGaussianMultiDim(ConstantGaussian):

    def forward(self, x):
        mean = torch.zeros((*[x.shape[:-1]], self.output_dim)).to(x)
        std = torch.ones((*[x.shape[:-1]], self.output_dim)).to(x) * self.std
        return Normal(loc=mean, scale=std)

    @property
    def output_size(self):
        return self.output_dim
