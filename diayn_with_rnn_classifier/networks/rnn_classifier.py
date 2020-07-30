import torch
import torch.nn as nn
import torch.nn.functional as F


from mode_disent.network.mode_model import BiRnn

from code_slac.network.base import BaseNetwork

from self_supervised.base.network.mlp import MyMlp

class SeqEncoder(BaseNetwork):
    """
    Encodes a sequence into a vector of skill_dim
    """
    def __init__(self,
                 skill_dim,
                 state_rep_dim,
                 hidden_rnn_dim,
                 hidden_units,
                 rnn_dropout,
                 num_rnn_layers
                 ):
        super(BaseNetwork, self).__init__()

        self.rnn = BiRnn(
            input_dim=state_rep_dim,
            hidden_rnn_dim=hidden_rnn_dim,
            rnn_layers=num_rnn_layers,
            rnn_dropout=rnn_dropout
        )

        self.net = MyMlp(
            input_size=hidden_rnn_dim * 2,
            output_size=skill_dim,
            hidden_sizes=hidden_units,
            output_activation=torch.nn.ReLU()
        )

    def forward(self,
                state_rep_seq):
        """
        Args:
            state_rep_seq        : (N, S, state_rep_dim)
        Return:
            vector               : (N, skill_dim)
        """
        assert state_rep_seq.size(-1) == self.rnn.input_dim

        rnn_out = self.rnn(state_rep_seq.transpose(0, 1))
        out = self.net(rnn_out)

        assert out.size(0) == state_rep_seq.size(1)
        return out


