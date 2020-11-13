import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from code_slac.network.latent import Gaussian, ConstantGaussian
from self_supervised.base.network.mlp import MyMlp, MlpWithDropout


class MyGaussian(Gaussian):

    def __init__(self,
                 input_dim,
                 output_dim,
                 calc_log_std=False,
                 hidden_units=None,
                 std=None,
                 leaky_slope=0.2,
                 dropout=0.,
                 layer_norm=False,
                 ):
        if hidden_units is None:
            super().__init__(
                input_dim=input_dim,
                output_dim=output_dim,
                std=std,
                leaky_slope=leaky_slope
            )
        else:
            super().__init__(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_units=hidden_units,
                std=std,
                leaky_slope=leaky_slope
            )

        # Overwrite base-net
        self.net = MlpWithDropout(
            input_size=input_dim,
            output_size=2*output_dim if std is None else output_dim,
            hidden_sizes=hidden_units,
            hidden_activation=nn.LeakyReLU(leaky_slope),
            dropout=dropout,
            layer_norm=layer_norm,
        )

        self.calc_log_std = calc_log_std

    @property
    def output_size(self):
        if self.std is None:
            assert self.net.output_size % 2 == 0

        _output_size = int(self.net.output_size/2) \
            if self.std is None \
            else self.net.output_size
        return _output_size

    def forward(self, x) -> Normal:
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.cat(x, dim=-1)

        x = self.net(x)
        if self.std:
            mean = x
            std = torch.ones_like(mean) * self.std

        elif not self.calc_log_std:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(std) + 1e-5

        else:
            mean, log_std = torch.chunk(x, 2, dim=-1)
            std = torch.exp(log_std)

        return Normal(loc=mean, scale=std)


class ConstantGaussianMultiDim(ConstantGaussian):

    def forward(self, x):
        mean = torch.zeros((*x.shape[:-1], self.output_dim)).to(x)
        std = torch.ones((*x.shape[:-1], self.output_dim)).to(x) * self.std
        return Normal(loc=mean, scale=std)

    @property
    def output_size(self):
        return self.output_dim


class ConstantGaussianMultiDimMeanSpec(ConstantGaussianMultiDim):

    def forward(self, mean):
        std = torch.ones((*mean.shape[:-1], self.output_dim)).to(mean) * self.std
        return Normal(loc=mean, scale=std)
