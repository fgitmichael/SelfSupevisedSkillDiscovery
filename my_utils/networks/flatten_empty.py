import torch

from code_slac.network.base import BaseNetwork


class FlattenEmpty(BaseNetwork):

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return flat_inputs
