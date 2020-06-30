import torch.nn as nn


class Empty(nn.Module):
    def forward(self, x):
        return x