import torch

from diayn_rnn_seq_rnn_stepwise_classifier.networks.positional_encoder import \
    PositionalEncoding as PositionalEncodingStacked


class PositionalEncodingTransformerStacked(PositionalEncodingStacked):

    def forward(self, x):
        """
        Args:
            x           : (N, S, data_dim)

        Return:
            x_pos       : (N, S, data_dim + data_dim)
        """
        to_cat = self.pe[:x.size(0), :]
        to_cat = torch.cat([to_cat] * x.size(1), dim=1)
        assert to_cat.shape == x.shape
        encoded = torch.cat([x, to_cat], dim=-1)

        return self.dropout(encoded)

