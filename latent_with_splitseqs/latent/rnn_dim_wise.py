import torch
import torch.nn as nn
import torch.nn.functional as F

from code_slac.network.base import BaseNetwork

from self_supervised.base.network.mlp import MyMlp, MlpWithDropout

from two_d_navigation_demo.base.create_posencoder import create_pos_encoder


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

        hidden_size_res = 2 * hidden_size if bidirectional else hidden_size
        self.feature_size_matcher = MlpWithDropout(
            input_size=input_size * hidden_size_res,
            output_size=out_feature_size,
            dropout=dropout,
        )
        self.normalizer = nn.LogSoftmax(dim=-1)

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
            hidden_seq, _ = self.rnn(one_dimenional_seq)
            hidden_out.append(hidden_seq)
        hidden_out_seq = torch.cat(hidden_out, dim=data_dim)
        assert hidden_out_seq.shape[:data_dim] == seq.shape[:-1]
        try:
            hidden_size_rnn_res = 2 * self.rnn.hidden_size if self.rnn.bidirectional else self.rnn.hidden_size
            assert hidden_out_seq.shape[-1] == hidden_size_rnn_res * seq.size(data_dim)
        except:
            raise ValueError

        matched_features = self.normalizer(self.feature_size_matcher(hidden_out_seq))

        if self.log_softmax_output:
            return F.log_softmax(matched_features, dim=data_dim), None

        else:
            return matched_features, None


class GRUDimwisePosenc(GRUDimwise):

    def __init__(self,
                 *args,
                 hidden_size,
                 bidirectional=False,
                 max_seqlen=1000,
                 **kwargs
                 ):
        super(GRUDimwisePosenc, self).__init__(
            *args,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            **kwargs
        )
        posenc_dict = create_pos_encoder(
            feature_dim=self.hidden_size,
            seq_len=max_seqlen,
            pos_encoder_variant='transformer',
        )
        self.pos_encoder = posenc_dict['pos_encoder']
        self.output_size = posenc_dict['pos_encoded_feature_dim']

    def forward(self, seq):
        """
        Args:
            seq             : (N, S, data_dim)
        Return:
            out             : (N, S, out_feature_size)
        """
        feature_seq, _ = super(GRUDimwisePosenc, self).forward(seq)
        feature_seq_posenc = self.pos_encoder(feature_seq)

        return feature_seq_posenc, None
