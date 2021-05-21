import torch
import torch.nn as nn
import torch.nn.functional as F

from code_slac.network.base import BaseNetwork

from mode_disent_no_ssm.utils.empty_network import Empty


class DfTransformer(BaseNetwork):

    def __init__(self,
                 input_size,
                 output_size=None,
                 num_heads=8,
                 num_layers=6,
                 ):
        super().__init__()
        embed_dim_multiple_num_heads = input_size % num_heads == 0
        if not embed_dim_multiple_num_heads:
            self.linear_input_adjust = nn.Linear(
                input_size,
                (input_size // num_heads) * num_heads + num_heads
            )
            input_size_adjusted = self.linear_input_adjust.out_features

        else:
            self.linear_input_adjust = Empty()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size_adjusted,
            nhead=num_heads,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        if output_size is not None:
            self.out_linear = nn.Linear(
                in_features=input_size_adjusted,
                out_features=output_size,
            )
            self.output_size = output_size

        elif not embed_dim_multiple_num_heads:
            # output_size is None as otherwise first case is used
            self.out_linear = nn.Linear(
                in_features=input_size_adjusted,
                out_features=input_size,
            )
            self.output_size = input_size

        else:
            # No output adjustment
            self.out_linear = Empty()
            self.output_size = input_size

    def forward(self, seq):
        """
        Args:
            seq                 : (N, S, data_dim)
        Returns:
            out                 : (N, S, output_size)
        """
        batch_dim, seq_dim, data_dim = 0, 1, 2

        # Input dimension adjust
        seq = self.linear_input_adjust(seq)

        # Transformer expects sbd-format
        seq_sbd = torch.transpose(seq, dim0=batch_dim, dim1=seq_dim)
        batch_dim, seq_dim, data_dim = 1, 0, 2

        # Apply networks
        transformer_out = self.transformer_encoder(seq_sbd)
        out_sbd = self.out_linear(transformer_out)

        # Transpose to bsd
        out = torch.transpose(out_sbd, dim0=batch_dim, dim1=seq_dim)
        batch_dim, seq_dim, data_dim = 0, 1, 2
        assert out.shape == torch.Size((
            seq.shape[batch_dim],
            seq.shape[seq_dim],
            self.output_size
        ))

        return out
