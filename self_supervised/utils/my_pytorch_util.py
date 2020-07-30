import torch
import rlkit.torch.pytorch_util as ptu

def rand(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = ptu.device
    return torch.rand(*args, **kwargs, device=torch_device)

def eye(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = ptu.device
    return torch.eye(*args, **kwargs, device=torch_device)
