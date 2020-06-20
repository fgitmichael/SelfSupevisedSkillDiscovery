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
                 rnn_dim,
                 num_rnn_layers,
                 rnn_dropout,
                 hidden_units_mode_encoder,
                 hidden_units_action_decoder,
                 mode_repeating,
                 feature_dim,
                 representation_dim,
                 action_dim,
                 std_decoder,
                 action_normalized,
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
        if mode_repeating:
            self.ac


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
                state_rep,
                mode_sample):
        """
        Args:
            state_rep       : (N, state_rep_dim) - tensor
            i                 Representation of state. Can be a latent variable,
                              list of latent variable or just a feature
            mode_sample     : (N, mode_dim) - tensor
        Return:
            action_dist     : Distribution over decoded actions
        """
        #Decode
        action_dist = self.net([state_rep, mode_sample])

        # Normalize action mean
        action_dist.loc = torch.tanh(action_dist.loc)

        return action_dist


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
                state_rep,
                mode_sample):
        """
        See ActionDecoder
        """
        mode_sample_repeated = torch.cat(self.mode_repeat * [mode_sample], dim=1)
        action_dist = super(ActionDecoderModeRepeat, self).forward(
            state_rep=state_rep,
            mode_sample=mode_sample_repeated
        )

        return action_dist
