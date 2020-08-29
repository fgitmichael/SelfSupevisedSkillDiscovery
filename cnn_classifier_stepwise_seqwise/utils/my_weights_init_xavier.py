import torch
from torch import nn

def weights_init_xavier(m):
    if isinstance(m, nn.Linear) \
            or isinstance(m, nn.Conv2d) \
            or isinstance(m, nn.Conv1d) \
            or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
