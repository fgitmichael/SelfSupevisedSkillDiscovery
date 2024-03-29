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

from seqwise_cont_skillspace.networks.nooh_encoder import NoohPosEncoder
from seqwise_cont_skillspace.networks.transformer_stack_pos_encoder import \
    PositionalEncodingTransformerStacked

from mode_disent_no_ssm.utils.empty_network import Empty


class RnnStepwiseSeqwiseClassifierBase(BaseNetwork, metaclass=abc.ABCMeta):

    def __init__(self,
                 obs_dim,
                 hidden_size_rnn,
                 skill_dim,
                 hidden_sizes: list,
                 seq_len,
                 normalize_before_feature_extraction: bool = False,
                 num_layers: int = 1,
                 dropout=0.0,
                 pos_encoder_variant='transformer'):
        """
        Args:
            obs_dim             : dimension of state representation
            hidden_size_rnn     : dimension of hidden state in the rnn
            skill_dim           : dimension of targets
        """
        super(RnnStepwiseSeqwiseClassifierBase, self).__init__()

        self.skill_dim = skill_dim
        self.rnn = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_size_rnn,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.rnn_params = {}
        self.rnn_params['num_directions'] = \
            2 if self.rnn.bidirectional else 1
        self.rnn_params['num_channels'] = \
            self.rnn_params['num_directions'] * self.rnn.num_layers
        self.rnn_params['num_features_h_n'] = \
            self.rnn_params['num_channels'] * self.rnn.hidden_size
        self.rnn_params['num_features_hidden_seq'] = \
            self.rnn_params['num_directions'] * self.rnn.hidden_size

        minimum_input_size_step_classifier = self.rnn_params['num_features_hidden_seq']
        pos_encode_dim = 0
        if pos_encoder_variant=='transformer':
            self.pos_encoder = PositionalEncoding(
                d_model=self.rnn_params['num_features_hidden_seq'],
                max_len=seq_len,
                dropout=0.1
            )
            input_size_classifier = minimum_input_size_step_classifier
            pos_encode_dim = 0

        elif pos_encoder_variant=='transformer_stacked':
            self.pos_encoder = PositionalEncodingTransformerStacked(
                d_model=self.rnn_params['num_features_hidden_seq'],
                max_len=seq_len,
                dropout=0.1
            )
            input_size_classifier = minimum_input_size_step_classifier * 2
            pos_encode_dim = minimum_input_size_step_classifier

        elif pos_encoder_variant=='oh_encoder':
            self.pos_encoder = PositionalEncodingOh()
            input_size_classifier = minimum_input_size_step_classifier + seq_len
            pos_encode_dim = seq_len

        elif pos_encoder_variant=='cont_encoder':
            self.pos_encoder = NoohPosEncoder(
                encode_dim=self.skill_dim,
                max_seq_len=300,
            )
            input_size_classifier = minimum_input_size_step_classifier + \
                                    self.pos_encoder.encode_dim
            pos_encode_dim = self.pos_encoder.encode_dim

        elif pos_encoder_variant=='empty':
            self.pos_encoder = Empty()
            input_size_classifier = minimum_input_size_step_classifier
            pos_encode_dim = 0

        else:
            raise NotImplementedError(
                "{} encoding is not implement".format(pos_encoder_variant))

        self.pos_encode_dim = pos_encode_dim
        self.rnn_params['num_features_hs_posenc'] = \
            self.rnn_params['num_features_hidden_seq'] + pos_encode_dim

        self.classifier_step = self.create_stepwise_classifier(
            feature_dim=input_size_classifier,
            skill_dim=self.skill_dim,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

        self.classifier_seq = self.create_seqwise_classifier(
            feature_dim=self.rnn_params['num_features_h_n'],
            skill_dim=self.skill_dim,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

        self.set_dimensions()

        self.normalize_before_feature_extraction = normalize_before_feature_extraction

    def set_dimensions(self):
        self.batch_dim = 0
        self.seq_dim = 1
        self.data_dim = -1

    @abc.abstractmethod
    def create_stepwise_classifier(
            self,
            feature_dim,
            skill_dim,
            hidden_sizes,
            dropout=0.
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def create_seqwise_classifier(
            self,
            feature_dim,
            skill_dim,
            hidden_sizes,
            dropout=0.,
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
        assert len(seq_batch.shape) == 3
        batch_size = seq_batch.size(self.batch_dim)
        seq_len = seq_batch.size(self.seq_dim)

        if self.normalize_before_feature_extraction:
            std, mean = torch.std_mean(seq_batch, dim=self.seq_dim)
            mean = torch.stack([mean] * seq_len, dim=self.seq_dim)
            std = torch.stack([std] * seq_len, dim=self.seq_dim)
            seq_batch = (seq_batch - mean)/(std + 1E-8)
            #std, mean = torch.std_mean(seq_batch, dim=self.seq_dim)

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

        h_n = h_n.transpose(1, 0)
        assert my_ptu.tensor_equality(
            h_n.reshape(
                batch_size,
                self.rnn_params['num_features_h_n'])[0][:self.rnn.hidden_size],
            h_n[0][0]
        )
        h_n = h_n.reshape(
            batch_size,
            self.rnn_params['num_features_h_n']
        )

        return hidden_seq, h_n

    @abc.abstractmethod
    def classify_stepwise(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def classify_seqwise(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self,
                seq_batch,
                train=False):
        raise NotImplementedError
