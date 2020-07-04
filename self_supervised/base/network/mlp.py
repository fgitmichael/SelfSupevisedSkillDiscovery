import torch
from torch import nn


import torch.nn.functional as F


from code_slac.network.base import weights_init_xavier


class MyMlp(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 initializer: weights_init_xavier,
                 hidden_activation=F.relu,
                 hidden_sizes: tuple = (256, 256),
                 output_activation=None,
                 layer_norm: bool = False):
        super(MyMlp, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm

        model = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            model.append(fc)

            if self.layer_norm:
                ln = nn.LayerNorm(next_size)
                model.append(ln)

            model.append(hidden_activation)

        last_fc = nn.Linear(in_size, output_size)
        model.append(last_fc)
        if output_activation is not None:
            model.append(output_activation)

        self.net = nn.Sequential(*model).apply(initializer)

    def forward(self, input: torch.Tensor):
        return self.net(input)