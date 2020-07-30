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
            state_rep_seq        : (S, N, state_rep_dim)
        Return:
            vector               : (N, skill_dim)
        """
        assert state_rep_seq.size(-1) == self.rnn.input_dim

        rnn_out = self.rnn(state_rep_seq)
        out = self.net(rnn_out)

        assert out.size(0) == state_rep_seq.size(1)
        return out


class SeqClassifierModule(object):

    def __init__(self,
                 encoder: SeqEncoder):
        self.encoder = encoder
        self.criterion = nn.CrossEntropyLoss()

    def loss_predictions(self, seq: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            seq              : (N, S, data_dim) tensor
            labels           : (N, 1) tensor
        Return:
            loss                    : scalar tensor
            prediction              : (N, skill_dim) tensor
            pediction_log_soft_max  : (N, skill_dim) tensor
        """
        batch_size = seq.size(0)
        assert labels.size() == torch.Size((batch_size, 1))

        pred = self.encoder(seq.transpose(0, 1))
        pred_log_softmax = F.log_softmax(pred, dim=1)

        loss = self.criterion(pred, labels.squeeze())

        return dict(
            loss=loss,
            prediction=pred,
            prediction_log_softmax=pred_log_softmax
        )
