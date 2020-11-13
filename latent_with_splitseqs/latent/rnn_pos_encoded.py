import torch
import torch.nn as nn

from code_slac.network.base import BaseNetwork

from self_supervised.base.network.mlp import MyMlp, MlpWithDropout

from two_d_navigation_demo.base.create_posencoder import create_pos_encoder


class GRUPosenc(BaseNetwork):

    def __init__(self,
                 input_size,
                 hidden_size,
                 dropout=0.,
                 bidirectional=False,
                 batch_first=True,
                 max_seqlen=1000,
                 ):
        super(GRUPosenc, self).__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

        posenc_dict = create_pos_encoder(
            feature_dim=2 * hidden_size if bidirectional else hidden_size,
            seq_len=max_seqlen,
            pos_encoder_variant='transformer',
        )

        self.pos_encoder = posenc_dict['pos_encoder']
        self.output_size = posenc_dict['pos_encoded_feature_dim']

    @property
    def hidden_size(self):
        return self.output_size

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
        batch_size, seq_len, input_data_dim = seq.shape

        hidden_seq, _ = self.rnn(seq)
        hidden_seq_posenc = self.pos_encoder(hidden_seq)
        assert hidden_seq_posenc.shape \
               == torch.Size((batch_size, seq_len, self.output_size))

        return hidden_seq_posenc, None
