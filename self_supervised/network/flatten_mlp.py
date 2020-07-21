import torch

from self_supervised.base.network.mlp import MyMlp


class FlattenMlp(MyMlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs)
