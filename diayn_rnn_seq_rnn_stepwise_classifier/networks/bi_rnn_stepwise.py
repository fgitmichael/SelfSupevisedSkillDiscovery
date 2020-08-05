import torch
from torch import nn

from code_slac.network.base import BaseNetwork
from self_supervised.network.flatten_mlp import FlattenMlp

from diayn_rnn_seq_rnn_stepwise_classifier.networks.positional_encoder import \
    PositionalEncoding
from diayn_rnn_seq_rnn_stepwise_classifier.networks.pos_encoder_oh import \
    PositionalEncodingOh


class BiRnnStepwiseClassifier(BaseNetwork):

    def __init__(self,
                 input_size,
                 hidden_size_rnn,
                 output_size,
                 hidden_sizes: list,
                 seq_len,
                 pos_encoder_variant='transformer',
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
        self.rnn_params = {}
        self.rnn_params['num_directions'] = \
            2 if self.rnn.bidirectional else 1
        self.rnn_params['num_channels'] = \
            self.rnn_params['num_directions'] * self.rnn.num_layers
        self.rnn_params['num_features'] = \
            self.rnn_params['num_channels'] * self.rnn.hidden_size

        minimal_input_size_classifier = self.rnn_params['num_features']
        if pos_encoder_variant=='transformer':
            self.pos_encoder = PositionalEncoding(
                d_model=self.rnn_params['num_features'],
                max_len=seq_len,
                dropout=0.1
            )
            input_size_classifier = minimal_input_size_classifier

        else:
            self.pos_encoder = PositionalEncodingOh()
            input_size_classifier = minimal_input_size_classifier + seq_len

        self.classifier = FlattenMlp(
            input_size=input_size_classifier,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
        )

    def _precess_seq(self, seq_batch):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = seq_batch.size(batch_dim)
        seq_len = seq_batch.size(seq_dim)
        hidden_seq, h_n = self.rnn(seq_batch)

        assert hidden_seq.shape == torch.Size(
            (batch_size,
             seq_len,
             self.rnn_params['num_directions'] * self.rnn.hidden_size)
        )

        return hidden_seq, h_n

    def _classify_seq_stepwise(self, hidden_seq):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = hidden_seq.size(batch_dim)
        seq_len = hidden_seq.size(seq_dim)
        data_dim = hidden_seq.size(data_dim)

        hidden_seq_pos_encoded = self.pos_encoder(hidden_seq)
        assert hidden_seq.shape == torch.Size(
            (batch_size,
             seq_len,
             self.rnn_params['num_features'])
        )

        classified = self.classifier(hidden_seq_pos_encoded)
        assert classified.shape == torch.Size(
            (batch_size,
             seq_len,
             self.classifier.output_size)
        )

        return classified

    def forward(self,
                seq_batch,
                return_rnn_outputs=False):
        """
        Args:
            seq_batch        : (N, S, data_dim)
        Return:
            d_pred           : (N * S, output_size)
        """
        assert len(seq_batch.shape) == 3

        hidden_seq, h_n = self._precess_seq(seq_batch)

        classified = self._classify_seq_stepwise(hidden_seq)

        if return_rnn_outputs:
            return classified, hidden_seq, h_n

        else:
            return classified
