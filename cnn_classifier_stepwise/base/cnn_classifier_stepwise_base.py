import abc
import torch
from operator import itemgetter

from code_slac.network.base import BaseNetwork

from cnn_classifier_stepwise.base.cnn_feature_extractor_base import \
    CnnFeatureExtractorBase

from two_d_navigation_demo.base.create_posencoder import \
    create_pos_encoder


class CnnStepwiseClassifierBaseDf(BaseNetwork, metaclass=abc.ABCMeta):

    def __init__(self,
                 skill_dim,
                 hidden_sizes_classifier_step,
                 seq_len,
                 feature_extractor: CnnFeatureExtractorBase,
                 pos_encoder_variant='transformer',
                 dropout=0.,
                 normalize_before_feature_extraction=False,
                 ):
        super().__init__()
        self.skill_dim = skill_dim
        self.feature_extractor = feature_extractor

        ret_dict_pos_encoder = create_pos_encoder(
            feature_dim=self.feature_extractor.output_feature_size,
            seq_len=seq_len,
            pos_encoder_variant=pos_encoder_variant,
            encoding_dim=2 if pos_encoder_variant=='cont_encoder' else None
        )

        self.pos_encoder, \
        pos_encoder_feature_dim = itemgetter(
            'pos_encoder',
            'pos_encoded_feature_dim')(ret_dict_pos_encoder)

        self.stepwise_classifier = self.create_stepwise_classifier(
            feature_dim=pos_encoder_feature_dim,
            skill_dim=self.skill_dim,
            hidden_sizes=hidden_sizes_classifier_step,
            dropout=dropout,
        )

        self.normalize_before_feature_extraction=normalize_before_feature_extraction

    @property
    @abc.abstractmethod
    def num_skills(self):
        raise NotImplementedError

    def _process_seq(self, seq):
        """
        Args:
            seq             : (N, S, data_dim)
        Return:
            feature_seq     : (N, S, feature_dim)
        """
        seq_dim = 1
        batch_size, seq_len, data_dim = seq.shape
        if self.normalize_before_feature_extraction:
            std, mean = torch.std_mean(seq, dim=seq_dim)
            mean = torch.stack([mean] * seq_len, dim=seq_dim)
            std = torch.stack([std] * seq_len, dim=seq_dim)
            seq = (seq - mean)/(std + 1E-8)

        return self.feature_extractor(seq)

    @abc.abstractmethod
    def create_stepwise_classifier(self,
                                   feature_dim,
                                   skill_dim,
                                   hidden_sizes,
                                   dropout=0.,
                                   ):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self,
                obs_next,
                train=False,
                ):
        raise NotImplementedError
