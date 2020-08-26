import torch
import torch.nn as nn


if __name__=="__main__":
    in_channels = 8
    out_channels = 10
    kernel_size = 3
    stride = 2
    batch_size = 4
    m = nn.Conv1d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride)
    input = torch.randn(batch_size, in_channels, 500)
    output = m(input)

    print(input.shape)
    print(output.shape)
    pass