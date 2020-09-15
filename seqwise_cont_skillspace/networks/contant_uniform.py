import torch
from torch import distributions

from code_slac.network.base import BaseNetwork


class ConstantUniform(BaseNetwork):

    def __init__(self, output_dim, low=-3., high=3.):
        super().__init__()
        self.output_dim = output_dim
        self.low = low
        self.high = high

    def forward(self, x):
        low = torch.ones((x.size(0), self.output_dim)).to(x) * self.low
        high = torch.ones((x.size(0), self.output_dim)).to(x) * self.high
        return distributions.Uniform(
            low=low,
            high=high,
        )


class ConstantUniformMultiDim(ConstantUniform):

    def forward(self, x:torch.Tensor):
        low = torch.ones((*x.shape[:-1], self.output_dim)).to(x) * self.low
        high = torch.ones((*x.shape[:-1], self.output_dim)).to(x) * self.high
        return distributions.Uniform(
            low=low,
            high=high
        )

    @property
    def output_size(self):
        return self.output_dim


