import torch
from torch import nn

from code_slac.network.base import BaseNetwork
from self_supervised.network.flatten_mlp import FlattenMlp


class BiRnnStepwiseClassifier(BaseNetwork):

    def __init__(self,
                 input_size,
                 hidden_size_rnn,
                 output_size,
                 hidden_sizes: list,
                 position_encoder: BaseNetwork
                 ):
        """
        Args:
            input_size        : dimension of state representation
            hidden_size_rnn   : dimension of hidden state in the rnn
            output_size       : dimension of targets
        """
        super(BiRnnStepwiseClassifier, self).__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size_rnn,
            batch_first=True,
            bidirectional=True
        )
        self.num_directions = 2 if self.rnn.bidirectional else 1

        self.pos_encoder = position_encoder

        self.classfier = FlattenMlp(
            input_size=self.num_directions * hidden_size_rnn,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            output_activation=nn.ReLU()
        )

    def forward(self, seq_batch):
        """
        Args:
            seq_batch        : (N, S, data_dim)
        Return:
            d_pred           : (N * S, output_size)
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = seq_batch.size(batch_dim)
        seq_len = seq_batch.size(seq_dim)
        data_dim = seq_batch.size(data_dim)
        assert len(seq_batch.shape) == 3

        hidden_seq, _ = self.rnn(seq_batch)
        assert hidden_seq.shape == torch.Size(
            (batch_size,
             seq_len,
             self.num_directions * self.rnn.hidden_size)
        )

        hidden_seq_pos_encoded = self.pos_encoder(hidden_seq)
        assert hidden_seq.shape == torch.Size(
            (batch_size,
             seq_len,
             self.num_directions * self.rnn.hidden_size)
        )

        classified = self.classfier(hidden_seq_pos_encoded)
        assert classified.shape == torch.Size(
            (batch_size,
             seq_len,
             self.classfier.output_size)
        )

        classified_out = classified.view(batch_size * seq_len, classified.size(data_dim))
        assert classified_out[:seq_len] == classified[0, :]

        return classified_out