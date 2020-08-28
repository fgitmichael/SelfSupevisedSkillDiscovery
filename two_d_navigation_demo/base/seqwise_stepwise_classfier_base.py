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


class SeqwiseStepwiseClassifierBase(BaseNetwork, metaclass=abc.ABCMeta):

    def __init__(self):
        super(SeqwiseStepwiseClassifierBase, self).__init__()
        self.set_dimensions()

    @abc.abstractmethod
    def _process_seq(self, seq_batch):
        raise NotImplementedError

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

    def set_dimensions(self):
        self.batch_dim = 0
        self.seq_dim = 1
        self.data_dim = -1

    @abc.abstractmethod
    def create_feature_extractor(self,
                                 **feature_extractor_params) -> dict:
        """
        Return:
            feature_extractor net
            feature_extractor parameter dict
        """
        raise NotImplementedError

    def create_pos_encoder(
            self,
            feature_dim: int,
            seq_len: int,
            pos_encoder_variant: str,
            encoding_dim=None
    ) -> dict:
        """
        Args:
            feature_dim
            seq_len
            pos_encoder_variant
            encoding_dim                : only used
                                        : if pos_encoder_varaint is 'cont_encoder'
        Return dict:
            pos_encoder                 : nn.Module
            pos_encoded_feature_dim     : int (= feature_dim + pos_encoding_dim)
        """
        return create_pos_encoder(
            feature_dim=feature_dim,
            seq_len=seq_len,
            pos_encoder_variant=pos_encoder_variant,
            encoding_dim=encoding_dim,
        )

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

def create_pos_encoder(
        feature_dim: int,
        seq_len: int,
        pos_encoder_variant: str,
        encoding_dim=None
) -> dict:
    """
    Args:
        feature_dim
        seq_len
        pos_encoder_variant
        encoding_dim                : only used
                                    : if pos_encoder_varaint is 'cont_encoder'
    Return dict:
        pos_encoder                 : nn.Module
        pos_encoded_feature_dim     : int (= feature_dim + pos_encoding_dim)
    """
    if encoding_dim is not None:
        assert pos_encoder_variant=='cont_encoder'

    minimum_input_size_step_classifier = feature_dim
    pos_encode_dim = 0
    if pos_encoder_variant=='transformer':
        pos_encoder = PositionalEncoding(
            d_model=minimum_input_size_step_classifier,
            max_len=seq_len,
            dropout=0.1
        )
        pos_encoded_feature_dim = minimum_input_size_step_classifier

    elif pos_encoder_variant=='transformer_stacked':
        pos_encoder = PositionalEncodingTransformerStacked(
            d_model=minimum_input_size_step_classifier,
            max_len=seq_len,
            dropout=0.1
        )
        pos_encoded_feature_dim = minimum_input_size_step_classifier * 2

    elif pos_encoder_variant=='oh_encoder':
        pos_encoder = PositionalEncodingOh()
        pos_encoded_feature_dim = minimum_input_size_step_classifier + seq_len

    elif pos_encoder_variant=='cont_encoder':
        pos_encoder = NoohPosEncoder(
            encode_dim=encoding_dim,
            max_seq_len=300,
        )
        pos_encoded_feature_dim = minimum_input_size_step_classifier + \
                                  pos_encoder.encode_dim

    elif pos_encoder_variant=='empty':
        pos_encoder = Empty()
        pos_encoded_feature_dim = minimum_input_size_step_classifier

    else:
        raise NotImplementedError(
            "{} encoding is not implement".format(pos_encoder_variant))

    return dict(
        pos_encoder=pos_encoder,
        pos_encoded_feature_dim=pos_encoded_feature_dim
    )
