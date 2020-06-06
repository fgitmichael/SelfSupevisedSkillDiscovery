import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

from code_slac.network.base import BaseNetwork
from code_slac.network.latent import Gaussian, ConstantGaussian


class ModeLatentNetwork(BaseNetwork):

    def __init__(self,
                 mode_dim,
                 rnn_dim,
                 num_rnn_layers,
                 hidden_units_mode_encoder,
                 hidden_units_action_decoder,
                 mode_repeating,
                 feature_dim,
                 action_dim,
                 dyn_latent_network,
                 std_decoder,
                 leaky_slope):
        super(ModeLatentNetwork, self).__init__()


        # Latent model for the dynamics
        self.dyn_latent_network = dyn_latent_network

        # Encoder net for mode q(m | x(1:T), a(1:T))
        self.mode_encoder = ModeEncoderCombined(
            feature_shape=feature_dim,
            output_dim=mode_dim,
            action_dim=action_dim,
            hidden_units=hidden_units_mode_encoder,
            hidden_rnn_dim=rnn_dim,
            rnn_layers=num_rnn_layers)

        # Mode prior
        self.mode_prior = ConstantGaussian(mode_dim)

        # Action decoder
        latent1_dim = self.dyn_latent_network.latent1_dim
        latent2_dim = self.dyn_latent_network.latent2_dim
        if mode_repeating:
            self.action_decoder = ActionDecoderModeRepeat(
                latent1_dim=latent1_dim,
                latent2_dim=latent2_dim,
                mode_dim=mode_dim,
                action_dim=action_dim,
                hidden_units=hidden_units_action_decoder,
                std=std_decoder,
                leaky_slope=leaky_slope)
        else:
            self.action_decoder = ActionDecoderNormal(
                latent1_dim=latent1_dim,
                latent2_dim=latent2_dim,
                mode_dim=mode_dim,
                action_dim=action_dim,
                hidden_units=hidden_units_action_decoder,
                std=std_decoder,
                leaky_slope=leaky_slope)

    def sample_mode_prior(self, batch_size):
        mode_dist = self.mode_prior(torch.rand(batch_size, 1))
        return {'mode_dist': mode_dist,
                'mode_sample': mode_dist.sample()
                }

    def sample_mode_posterior(self, features_seq, actions_seq):
        mode_dist = self.mode_encoder(features_seq=features_seq,
                                      actions_seq=actions_seq)
        mode_sample = mode_dist.rsample()
        return {'mode_dist': mode_dist,
                'mode_samples': mode_sample}


class BiRnn(BaseNetwork):

    def __init__(self,
                 input_dim,
                 hidden_rnn_dim,
                 rnn_layers=1,
                 learn_initial_state=True):
        super(BiRnn, self).__init__()

        # RNN
        # Note: batch_first=True means input and output dims are treated as
        #       (batch, seq, feature)
        self.input_dim = input_dim
        self.hidden_rnn_dim = hidden_rnn_dim
        self.f_rnn = nn.GRU(self.input_dim, self.hidden_rnn_dim,
                            num_layers=rnn_layers,
                            bidirectional=True)

        # Noisy hidden init state
        # Note: Only works with GRU right now
        self.learn_init = learn_initial_state
        if self.learn_init:
            # Initial state (dim: num_layers * num_directions, batch, hidden_size)
            self.init_network = Gaussian(input_dim=1,
                                         output_dim=self.f_rnn.hidden_size,
                                         hidden_units=[256])

    def forward(self, x):
        num_sequence = x.size(0)
        batch_size = x.size(1)

        # Initial state (dim: num_layers * num_directions, batch, hidden_size)
        if self.learn_init:
            num_directions = 2 if self.f_rnn.bidirectional else 1
            init_input = torch.ones(self.f_rnn.num_layers * num_directions,
                                    batch_size,
                                    1).to(x.device)
            hidden_init = self.init_network(init_input).rsample()

            # LSTM recursion and extraction of the ends of the two directions
            # (front: end of the forward pass, back: end of the backward pass)
            rnn_out, _ = self.f_rnn(x, hidden_init)
        else:
            # Don't use learned initial state and rely on pytorch init
            rnn_out, _ = self.f_rnn(x)

        # Split into the two directions
        (forward_out, backward_out) = torch.chunk(rnn_out, 2, dim=2)

        # Get the ends of the two directions
        front = forward_out[num_sequence - 1, :, :]
        back = backward_out[0, :, :]

        # Stack along hidden_dim and return
        return torch.cat([front, back], dim=1)


# TODO: Move this class as inner class to ModeDisentanglingNetwork as it is
#      too sophisticated
class ModeEncoder(BaseNetwork):

    def __init__(self,
                 feature_shape,
                 action_shape,
                 output_dim,  # typically mode_dim
                 hidden_rnn_dim,
                 hidden_units,
                 rnn_layers
                 ):
        super(ModeEncoder, self).__init__()

        self.f_rnn_features = BiRnn(feature_shape,
                                    hidden_rnn_dim=hidden_rnn_dim,
                                    rnn_layers=rnn_layers)
        self.f_rnn_actions = BiRnn(action_shape,
                                   hidden_rnn_dim=hidden_rnn_dim,
                                   rnn_layers=rnn_layers)

        # Concatenation of 2*hidden_rnn_dim from the features rnn and
        # 2*hidden_rnn_dim from actions rnn, hence input dim is 4*hidden_rnn_dim
        self.f_dist = Gaussian(input_dim=4 * hidden_rnn_dim,
                               output_dim=output_dim,
                               hidden_units=hidden_units)

    def forward(self, features_seq, actions_seq):
        feat_res = self.f_rnn_features(features_seq)
        act_res = self.f_rnn_actions(actions_seq)
        rnn_result = torch.cat([feat_res, act_res], dim=1)

        # Feed result into Gaussian layer
        return self.f_dist(rnn_result)


class ModeEncoderCombined(BaseNetwork):

    def __init__(self,
                 feature_shape,
                 action_dim,
                 output_dim,  # typicall mode_dim
                 hidden_rnn_dim,
                 hidden_units,
                 rnn_layers):
        super(BaseNetwork, self).__init__()

        self.rnn = BiRnn(feature_shape + action_dim,
                         hidden_rnn_dim=hidden_rnn_dim,
                         rnn_layers=rnn_layers)

        self.mode_dist = Gaussian(input_dim=2 * hidden_rnn_dim,
                                  output_dim=output_dim,
                                  hidden_units=hidden_units)

    def forward(self, features_seq, actions_seq):
        # State-seq-len is always shorter by one than action_seq_len, but to stack the
        # sequence need to have the same length. Solution: learn the missing element in the state seq
        # Solution: discard last action
        assert features_seq.size(0) + 1 == actions_seq.size(0)
        actions_seq = actions_seq[:-1, :, :]

        seq = torch.cat([features_seq, actions_seq], dim=2)

        rnn_result = self.rnn(seq)

        return self.mode_dist(rnn_result)


class ActionDecoderModeRepeat(BaseNetwork):

    def __init__(self,
                 latent1_dim,
                 latent2_dim,
                 mode_dim,
                 action_dim,
                 hidden_units,
                 leaky_slope,
                 std=None):
        super(ActionDecoderModeRepeat, self).__init__()

        latent_dim = latent1_dim + latent2_dim
        if latent_dim > mode_dim:
            self.mode_repeat = 10 * latent_dim//mode_dim
        else:
            self.mode_repeat = 1

        self.net = Gaussian(latent_dim+self.mode_repeat*mode_dim,
                            action_dim,
                            hidden_units=hidden_units,
                            leaky_slope=leaky_slope,
                            std=std)

    def forward(self,
                latent1_sample,
                latent2_sample,
                mode_sample):
        assert len(latent1_sample.shape) \
               == len(latent2_sample.shape) \
               == len(mode_sample.shape)
        mode_sample_input = torch.cat(self.mode_repeat * [mode_sample], dim=-1)
        net_input = torch.cat([latent1_sample, latent2_sample, mode_sample_input], dim=-1)
        action_dist = self.net(net_input)

        return action_dist


class ActionDecoderNormal(BaseNetwork):

    def __init__(self,
                 latent1_dim,
                 latent2_dim,
                 mode_dim,
                 action_dim,
                 hidden_units,
                 leaky_slope,
                 std=None):
        super(ActionDecoderNormal, self).__init__()

        self.net = Gaussian(
            input_dim=latent1_dim + latent2_dim + mode_dim,
            output_dim=action_dim,
            hidden_units=hidden_units,
            std=std,
            leaky_slope=leaky_slope)

    def forward(self,
                latent1_sample,
                latent2_sample,
                mode_sample):
        return self.net([latent1_sample, latent2_sample, mode_sample])
