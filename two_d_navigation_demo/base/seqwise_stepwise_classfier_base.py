import abc

from code_slac.network.base import BaseNetwork

from two_d_navigation_demo.base.create_posencoder import create_pos_encoder


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

