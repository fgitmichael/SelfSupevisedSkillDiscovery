import torch
import torch.nn as nn
import torch.nn.functional as F

from code_slac.network.base import BaseNetwork

from self_supervised.base.network.mlp import MyMlp, MlpWithDropout

class GRUDimwise(BaseNetwork):

    def __init__(self,
                 input_size,
                 hidden_size,
                 out_feature_size,
                 dropout=0.,
                 log_softmax_output=False,
                 bidirectional=False,
                 batch_first=True,
                 ):
        super(GRUDimwise, self).__init__()

        self.rnn = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

        self.feature_size_matcher = MlpWithDropout(
            input_size=input_size,
            output_size=out_feature_size,
            dropout=dropout,
        )

        self.log_softmax_output = log_softmax_output

    @property
    def hidden_size(self):
        return self.feature_size_matcher.output_size

    def forward(self, seq):
        """
        Args:
            seq             : (N, S, data_dim)
        Return:
            out             : (N, S, out_feature_size)
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        seq_permuted = seq.permute(data_dim, batch_dim, seq_dim)

        hidden_out = []
        for one_dimenional_seq in seq_permuted:
            one_dimenional_seq = one_dimenional_seq.unsqueeze(data_dim)
            hidden_out.append(self.rnn(one_dimenional_seq))
        hidden_out_seq = torch.cat(hidden_out, dim=data_dim)
        assert hidden_out_seq.shape[:-1] == seq.shape[:-1]
        assert hidden_out_seq.shape[-1] == self.rnn.hidden_size * seq.size(data_dim)

        matched_features = self.feature_size_matcher(hidden_out_seq)

        if self.log_softmax_output:
            return F.log_softmax(matched_features, dim=data_dim)

        else:
            return matched_features
