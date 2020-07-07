import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu

def my_from_numpy(*args, **kwargs):
    for el in args:
        yield ptu.from_numpy(el)

    for k, v in kwargs:
        kwargs[k] = ptu.from_numpy(v)

    yield kwargs

