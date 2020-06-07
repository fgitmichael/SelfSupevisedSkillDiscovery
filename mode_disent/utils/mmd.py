import torch


# Code from https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder
def compute_kernel_tutorial(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)

# Code from https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder
def compute_mmd_tutorial(x, y):
    assert x.shape == y.shape
    x_kernel = compute_kernel_tutorial(x, x)
    y_kernel = compute_kernel_tutorial(y, y)
    xy_kernel = compute_kernel_tutorial(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd