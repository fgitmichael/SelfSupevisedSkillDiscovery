import torch
from torch import distributions


def reshape_normal(dist, *args, **kwargs):
    loc = dist.loc.reshape(*args, **kwargs)
    scale = dist.scale.reshape(*args, **kwargs)

    return distributions.Normal(
        loc=loc,
        scale=scale,
    )