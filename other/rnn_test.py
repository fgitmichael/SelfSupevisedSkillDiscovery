import torch
from torch import nn

feature_size = 5
hidden_size = 10

mygru = nn.GRU(
    input_size=feature_size,
    hidden_size=hidden_size,
    batch_first=True,
    bidirectional=True
)
mygru_forward = nn.GRU(
    input_size=feature_size,
    hidden_size=hidden_size,
    batch_first=True,
)
mygru_backward = nn.GRU(
    input_size=feature_size,
    hidden_size=hidden_size,
    batch_first=True,
)

mygru.weight_hh_l0 = mygru_forward.weight_hh_l0
mygru.weight_ih_l0 = mygru_forward.weight_ih_l0
mygru.bias_hh_l0 = mygru_forward.bias_hh_l0
mygru.bias_ih_l0 = mygru_forward.bias_ih_l0

mygru_backward.weight_hh_l0 = mygru.weight_hh_l0_reverse
mygru_backward.weight_ih_l0 = mygru.weight_ih_l0_reverse
mygru_backward.bias_hh_l0 = mygru.bias_hh_l0_reverse
mygru_backward.bias_ih_l0 = mygru.bias_ih_l0_reverse

batch_size = 16
seq_len = 100
test_seq = torch.rand(batch_size,
                      seq_len,
                      feature_size)

out_tuple_bi = mygru(test_seq)
output_bi = out_tuple_bi[0]
h_n_bi = out_tuple_bi[1]

h_n_bi_chunks = torch.chunk(h_n_bi, chunks=2, dim=0)

output_bi_start_chunks = torch.chunk(output_bi[:, 0, :], chunks=2, dim=-1)
output_bi_chunks_end = torch.chunk(output_bi[:, -1, :], chunks=2, dim=-1)

assert torch.all(h_n_bi_chunks[0] == output_bi_chunks_end[0])
assert torch.all(h_n_bi_chunks[-1] == output_bi_start_chunks[-1])

reverse_idx = torch.arange(seq_len-1, -1, -1)
out_backward = mygru_backward(test_seq[:, reverse_idx, :])[0]
out_forward = mygru_forward(test_seq)[0]

out_forward_bi = torch.chunk(out_tuple_bi[0], chunks=2, dim=-1)[0]
out_backward_bi = torch.chunk(out_tuple_bi[0], chunks=2, dim=-1)[-1]

assert torch.all(out_forward_bi == out_forward)
assert torch.all(out_backward_bi[:, reverse_idx, :] == out_backward)
