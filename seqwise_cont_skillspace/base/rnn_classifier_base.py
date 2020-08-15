import torch
from torch import nn
from torch.nn import functional as F
import abc

from code_slac.network.base import BaseNetwork

from self_supervised.network.flatten_mlp import FlattenMlp
import self_supervised.utils.my_pytorch_util as my_ptu

from diayn_rnn_seq_rnn_stepwise_classifier.networks.positional_encoder import \
    PositionalEncoding
from diayn_rnn_seq_rnn_stepwise_classifier.networks.pos_encoder_oh import \
    PositionalEncodingOh


class StepwiseSeqwiseClassifierBase(BaseNetwork, metaclass=abc.ABCMeta):

    def __init__(self,
                 input_size,
                 hidden_size_rnn,
                 output_size,
                 hidden_sizes: list,
                 seq_len,
                 pos_encoder_variant='transformer'):
        """
        Args:
            input_size        : dimension of state representation
            hidden_size_rnn   : dimension of hidden state in the rnn
            output_size       : dimension of targets
        """
        super(StepwiseSeqwiseClassifierBase, self).__init__()

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

        self.step_classifier = self.create_stepwise_classifier(
            input_size=input_size_classifier,
            output_size=output_size,
            hidden_sizes=hidden_sizes
        )

        self.seq_classifier = self.create_seqwise_classifier(
            input_size=self.rnn_params['num_features'],
            output_size=output_size,
            hidden_sizes=hidden_sizes,
        )

        self.set_dimensions()

    def set_dimensions(self):
        self.batch_dim = 0
        self.seq_dim = 1
        self.data_dim = -1

    @abc.abstractmethod
    def create_stepwise_classifier(
            self,
            input_size,
            output_size,
            hidden_sizes
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def create_seqwise_classifier(
            self,
            input_size,
            output_size,
            hidden_sizes
    ):
        raise NotImplementedError

    def _process_seq(self, seq_batch):
        """
        Args:
            seq_batch           : (N, S, data_dim)
        Return:
            hidden_seq          : (N, S, 2 * hidden_size_rnn) if bidirectional
            h_n                 : (N, num_features)
        """
        batch_size = seq_batch.size(self.batch_dim)
        seq_len = seq_batch.size(self.seq_dim)

        hidden_seq, h_n = self.rnn(seq_batch)
        assert hidden_seq.shape == torch.Size(
            (batch_size,
             seq_len,
             self.rnn_params['num_directions'] * self.rnn.hidden_size)
        )
        assert h_n.shape == torch.Size(
            (self.rnn_params['num_channels'],
             batch_size,
             self.rnn.hidden_size)
        )

        assert my_ptu.tensor_equality(
            h_n.transpose(1, 0).reshape(
                batch_size,
                self.rnn_params['num_features'])[0][self.rnn.hidden_size:],
            h_n.transpose(1, 0)[0][0]
        )
        h_n = h_n.transpose(1, 0)
        h_n = h_n.reshape(
            batch_size,
            self.rnn_params['num_features']
        )

        return hidden_seq, h_n

    @abc.abstractmethod
    def forward(self,
                seq_batch,
                train=False):
        raise NotImplementedError
