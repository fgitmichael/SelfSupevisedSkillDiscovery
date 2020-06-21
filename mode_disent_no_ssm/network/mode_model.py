import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal
import abc

from code_slac.network.base import BaseNetwork
from code_slac.network.latent import Gaussian, ConstantGaussian
from mode_disent.network.mode_model import BiRnn


# Note: Normalized Actions are assumed
class ModeLatentNetwork(BaseNetwork):

    def __init__(self,
                 mode_dim,
                 representation_dim,
                 rnn_dim,
                 num_rnn_layers,
                 rnn_dropout,
                 hidden_units_mode_encoder,
                 hidden_units_action_decoder,
                 num_mode_repeat: int,
                 feature_dim,
                 action_dim,
                 std_decoder,
                 device,
                 leaky_slope
                 ):

        super(ModeLatentNetwork, self).__init__()

        self.device = device

        # Mode posterior
        self.mode_encoder = ModeEncoderFeaturesOnly(
            feature_dim=feature_dim,
            mode_dim=mode_dim,
            hidden_rnn_dim=rnn_dim,
            hidden_units=hidden_units_mode_encoder,
            rnn_dropout=rnn_dropout,
            num_rnn_layers=num_rnn_layers,
        )

        # Mode prior
        self.mode_prior = ConstantGaussian(mode_dim)

        # Action decoder
        if not num_mode_repeat == 0:
            self.action_decoder = ActionDecoderModeRepeat(
                state_rep_dim=representation_dim,
                mode_dim=mode_dim,
                action_dim=action_dim,
                hidden_units=hidden_units_action_decoder,
                std=std_decoder,
                leaky_slope=leaky_slope,
                num_mode_repeat=num_mode_repeat,
            )

        else:
            self.action_decoder = ActionDecoder(
                state_rep_dim=representation_dim,
                mode_dim=mode_dim,
                action_dim=action_dim,
                hidden_units=hidden_units_action_decoder,
                std=std_decoder,
                leaky_slope=leaky_slope,
            )

    def sample_mode_prior(self, batch_size):
        mode_dist = self.mode_prior(torch.rand(batch_size, 1).to(self.device))
        return {'dists': mode_dist,
                'samples': mode_dist.sample()}

    def sample_mode_posterior(self,
                              features_seq):
        """
        Args:
            features_seq    : (N, S + 1, feature_dim) tensor
        """
        features_seq = features_seq.transpose(0, 1)

        mode_dist = self.mode_encoder(features_seq=features_seq)
        mode_sample = mode_dist.rsample()

        return {'dists': mode_dist,
                'samples': mode_sample}


class ModeEncoderFeaturesOnly(BaseNetwork):

    def __init__(self,
                 feature_dim,
                 mode_dim,
                 hidden_rnn_dim,
                 hidden_units,
                 rnn_dropout,
                 num_rnn_layers
                 ):
        super(BaseNetwork, self).__init__()

        self.rnn = BiRnn(input_dim=feature_dim,
                         hidden_rnn_dim=hidden_rnn_dim,
                         rnn_layers=num_rnn_layers,
                         rnn_dropout=rnn_dropout
                         )

        self.mode_dist = Gaussian(input_dim=2 * hidden_rnn_dim,
                                  output_dim=mode_dim,
                                  hidden_units=hidden_units)

    def forward(self, features_seq):
        """
        Args:
            features_seq    : (S, N, feature_dim) tensor
        Return:
            mode_dist       : (N, mode_dim) distribution
        """
        rnn_out = self.rnn(features_seq)
        return self.mode_dist(rnn_out)


# TODO: implement a abstract baseclass for ActionDecoder (name ActionDecoderBase)
# Note: This Decoder Network returns normalized Actions!
class ActionDecoder(BaseNetwork):

    def __init__(self,
                 state_rep_dim,
                 mode_dim,
                 action_dim,
                 hidden_units,
                 leaky_slope,
                 std=None
                 ):
        super(ActionDecoder, self).__init__()

        self.net = Gaussian(
            input_dim=state_rep_dim + mode_dim,
            output_dim=action_dim,
            hidden_units=hidden_units,
            std=std,
            leaky_slope=leaky_slope
        )

    def forward(self,
                state_rep_seq,
                mode_sample: torch.tensor):
        """
        Args:
            state_rep_seq   : (N, S + 1, state_rep_dim) - tensor
                              Representation of state. Can be a latent variable,
                              list of latent variable or just a feature
            mode_sample     : (N, S + 1, mode_dim) - tensor
        Return:
            action_dist     : (N, S + 1) Distribution over decoded actions
        """
        # Repeat mode samples
        seq_len = state_rep_seq.size(1)
        mode_samples_repetition = torch.stack(seq_len * [mode_sample], dim=1)

        # Decode
        action_dist = self.net([state_rep_seq, mode_samples_repetition])

        # Normalize action mean
        action_dist.loc = torch.tanh(action_dist.loc)

        return {'dists': action_dist,
                'samples': action_dist.loc}


class ActionDecoderModeRepeat(ActionDecoder):

    def __init__(self,
                 state_rep_dim,
                 mode_dim,
                 action_dim,
                 hidden_units,
                 leaky_slope,
                 num_mode_repeat,
                 std=None,
                 ):
        if state_rep_dim > mode_dim:
            self.mode_repeat = num_mode_repeat * state_rep_dim//mode_dim
        else:
            self.mode_repeat = 1

        super(ActionDecoderModeRepeat, self).__init__(
            state_rep_dim=state_rep_dim,
            mode_dim=self.mode_repeat * mode_dim,
            action_dim=action_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope,
            std=std
        )

    def forward(self,
                state_rep_seq,
                mode_sample: torch.tensor):
        """
        Args:
            state_rep_seq   : (N, S + 1, state_rep_dim) - tensor
                              Representation of state. Can be a latent variable,
                              list of latent variable or just a feature
            mode_sample     : (N, S + 1, mode_dim) - tensor
        Return:
            action_dist     : (N, S + 1) Distribution over decoded actions
        """
        mode_sample_repeated = torch.cat(self.mode_repeat * [mode_sample], dim=1)
        actions = super(ActionDecoderModeRepeat, self).forward(
            state_rep_seq=state_rep_seq,
            mode_sample=mode_sample_repeated
        )

        return actions
