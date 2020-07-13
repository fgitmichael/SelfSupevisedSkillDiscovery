import torch
import torch.nn as nn
from typing import Dict

import rlkit.torch.pytorch_util as ptu

from code_slac.network.base import BaseNetwork
from code_slac.network.latent import Gaussian, ConstantGaussian

from mode_disent_no_ssm.network.mode_model import ModeEncoderFeaturesOnly

class ModeEncoderSelfSupComb(BaseNetwork):

    def __init__(self,
                 feature_dim,
                 mode_dim,
                 rnn_dim,
                 hidden_units,
                 rnn_dropout,
                 num_rnn_layers,
                 ):

        super().__init__()

        self.mode_encoder = ModeEncoderFeaturesOnly(
            feature_dim=feature_dim,
            mode_dim=mode_dim,
            hidden_rnn_dim=rnn_dim,
            hidden_units=hidden_units,
            rnn_dropout=rnn_dropout,
            num_rnn_layers=num_rnn_layers
        )

        self.mode_prior = ConstantGaussian(mode_dim)

    def forward(self, features_seq: torch.Tensor) -> dict:
        """
        Args:
            features_seq        : (N, S, feature_dim) tensor
        Return:
            post
                dist            : (N, mode_dim) distributions
                samples         : (N, mode_dim) tensor of samples
            pri
                dist            : (N, mode_dim) distributions
                samples         : (N, mode_dim) tensor of samples
        """
        batch_dim = 0
        seq_dim = 1
        batch_size = features_seq.size(batch_dim)

        assert len(features_seq.shape) == 3

        # Posterior
        features_seq = features_seq.transpose(seq_dim, batch_dim)
        post_mode_dist = self.mode_encoder(features_seq=features_seq)
        post_mode_samples = post_mode_dist.rsample()
        post = {
            'dist': post_mode_dist,
            'sample': post_mode_samples
        }

        # Prior
        pri_mode_dist = self.mode_prior(ptu.randn(batch_size, 1))
        pri_mode_samples = pri_mode_dist.sample()
        pri = {
            'dist': pri_mode_dist,
            'samples': pri_mode_samples
        }

        return {
            'post': post,
            'pri': pri
        }










