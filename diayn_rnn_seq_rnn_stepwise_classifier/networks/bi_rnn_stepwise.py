import torch
from torch import nn

from code_slac.network.base import BaseNetwork
from self_supervised.network.flatten_mlp import FlattenMlp, FlattenMlpDropout

from diayn_rnn_seq_rnn_stepwise_classifier.networks.positional_encoder import \
    PositionalEncoding
from diayn_rnn_seq_rnn_stepwise_classifier.networks.pos_encoder_oh import \
    PositionalEncodingOh

from mode_disent_no_ssm.utils.empty_network import Empty


class BiRnnStepwiseClassifier(BaseNetwork):

    def __init__(self,
                 input_size,
                 hidden_size_rnn,
                 output_size,
                 hidden_sizes: list,
                 seq_len,
                 obs_dims_used: tuple = None,
                 normalize_before_feature_extraction: bool = False,
                 dropout=0.,
                 pos_encoder_variant='empty',
                 num_layers=1,
                 bias=True,
                 layer_norm=False,
                 ):
        """
        Args:
            input_size        : dimension of state representation
            hidden_size_rnn   : dimension of hidden state in the rnn
            output_size       : dimension of targets
        """
        super(BiRnnStepwiseClassifier, self).__init__()

        if obs_dims_used is not None:
            input_size = len(obs_dims_used)
            self.obs_dims_used = obs_dims_used

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size_rnn,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.,
            bias=bias,
        )
        self.rnn_params = {}
        self.rnn_params['num_directions'] = \
            2 if self.rnn.bidirectional else 1
        self.rnn_params['num_channels'] = \
            self.rnn_params['num_directions'] * self.rnn.num_layers
        self.rnn_params['num_features_h_n'] = \
            self.rnn_params['num_channels'] * self.rnn.hidden_size
        self.rnn_params['num_features_hidden_seq'] = \
            self.rnn.hidden_size * self.rnn_params['num_directions']

        minimal_input_size_classifier = self.rnn_params['num_features_hidden_seq']
        if pos_encoder_variant=='transformer':
            self.pos_encoder = PositionalEncoding(
                d_model=self.rnn_params['num_features_hidden_seq'],
                max_len=seq_len,
                dropout=0.1
            )
            input_size_classifier = minimal_input_size_classifier

        elif pos_encoder_variant=='empty':
            self.pos_encoder = Empty()
            input_size_classifier = minimal_input_size_classifier
        else:
            raise NotImplementedError

        self.classifier = self.create_classifier(
            input_size=input_size_classifier,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            layer_norm=layer_norm,
        )

        self.normalize_before_feature_extraction = normalize_before_feature_extraction
        self.obs_dims_used = obs_dims_used

    def create_classifier(self,
                          input_size,
                          output_size,
                          hidden_sizes,
                          dropout,
                          layer_norm=False,
                          ):
        return FlattenMlpDropout(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            layer_norm=False,
            dropout=dropout,
        )

    def _process_seq(self, seq_batch):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = seq_batch.size(batch_dim)
        seq_len = seq_batch.size(seq_dim)

        if self.obs_dims_used is not None:
            seq_batch = seq_batch[:, :, self.obs_dims_used]

        if self.normalize_before_feature_extraction:
            std, mean = torch.std_mean(seq_batch, dim=seq_dim)
            mean = torch.stack([mean] * seq_len, dim=seq_dim)
            std = torch.stack([std] * seq_len, dim=seq_dim)
            seq_batch = (seq_batch - mean)/(std + 1E-8)

        hidden_seq, h_n = self.rnn(seq_batch)
        assert hidden_seq.shape == torch.Size(
            (batch_size,
             seq_len,
             self.rnn_params['num_directions'] * self.rnn.hidden_size)
        )

        return hidden_seq, h_n

    def _classify_stepwise(self, hidden_seq):
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
             self.rnn_params['num_features_hidden_seq'])
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

        hidden_seq, h_n = self._process_seq(seq_batch)

        classified = self._classify_stepwise(hidden_seq)

        if return_rnn_outputs:
            return classified, hidden_seq, h_n

        else:
            return classified
