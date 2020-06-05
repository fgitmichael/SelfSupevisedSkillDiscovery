import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

from code.network.base import BaseNetwork
from code.network.latent import Gaussian, ConstantGaussian, Encoder, Decoder
from .my_base import EncoderStateRep

class DynLatentNetwork(BaseNetwork):

    def __init__(self,
                 observation_shape,
                 action_shape,
                 feature_dim,
                 latent1_dim,
                 latent2_dim,
                 hidden_units,
                 hidden_units_encoder,
                 hidden_units_decoder,
                 leaky_slope,
                 state_rep):
        super(DynLatentNetwork, self).__init__()

        # p(z1(0)) = N(0, I)
        self.latent1_init_prior = ConstantGaussian(latent1_dim)
        # p(z2(0) | z1(0))
        self.latent2_init_prior = Gaussian(
            input_dim=latent1_dim,
            output_dim=latent2_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope)

        # p(z1(t+1) | z2(t), a(t), x_gt(t-1))
        self.latent1_prior = Gaussian(
            input_dim=latent2_dim + action_shape[0] + feature_dim,
            output_dim=latent1_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope)
        # p(z2(t+1) | z1(t+1), z2(t), a(t), x_gt(t-1))
        self.latent2_prior = Gaussian(
            input_dim=latent1_dim + latent2_dim + action_shape[0],
            output_dim=latent2_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope)

        # q(z1(0) | feat(0))
        self.latent1_init_posterior = Gaussian(
            input_dim=feature_dim,
            output_dim=latent1_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.latent2_init_posterior = self.latent2_init_prior

        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.latent1_posterior = Gaussian(
            input_dim=feature_dim + latent2_dim + action_shape[0],
            output_dim=latent1_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope)
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_posterior = self.latent2_prior

        # feat(t) = x(t) : This encoding is performed deterministically.
        if state_rep:
            # State representation
            self.encoder = EncoderStateRep(observation_shape[0],
                                           observation_shape[0],
                                           hidden_units=hidden_units_encoder)
        else:
            # Conv-nets for pixel observations
            self.encoder = Encoder(observation_shape[0],
                                   feature_dim,
                                   leaky_slope=leaky_slope)

        # p(x(t) | z1(t), z2(t))
        if state_rep:
            self.decoder = Gaussian(
                latent1_dim + latent2_dim,
                observation_shape[0],
                std=np.sqrt(0.1),
                hidden_units=hidden_units_decoder,
                leaky_slope=leaky_slope
            )
        else:
            self.decoder = Decoder(
                latent1_dim + latent2_dim,
                observation_shape[0],
                std=np.sqrt(0.1),
                leaky_slope=leaky_slope
            )

        # Dimensions
        self.latent1_dim = latent1_dim
        self.latent2_dim = latent2_dim

    def sample_prior_train(self, actions_seq, feature_seq):
        """
        Args:
            actions_seq        : (N, S, *action_shape)
            feature_seq        : (N, S+1, *feature_shape)
        Returns:
            latent1_samples    : (N, S+1, latent1_dim)
            latent2_samples    : (N, S+1, latent2_dim)
            latent1_dists      : (S+1) length list of (N, latent1_dim) distributions
            latent2_dists      : (S+1) length list of (N, latent2_dim) distributions
        """
        num_sequences = actions_seq.size(1)
        actions_seq = torch.transpose(actions_seq, 0, 1)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(num_sequences + 1):
            if t == 0:
                latent1_dist = self.latent1_init_prior(actions_seq[t])
                latent1_sample = latent1_dist.rsample()

                latent2_dist = self.latent2_init_prior(latent1_sample)
                latent2_sample = latent2_dist.rsample()

            else:
                latent1_dist = self.latent1_prior(
                    [latent2_samples[t-1], actions_seq[t-1], feature_seq[t-1]])
                latent1_sample = latent1_dist.rsample()

                latent2_dist = self.latent2_prior(
                    [latent1_sample, latent2_samples[t-1], actions_seq[t-1]])
                latent2_sample = latent2_dist.rsample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        return {'latent1_samples': latent1_samples,
                'latent2_samples': latent2_samples,
                'latent1_dists': latent1_dists,
                'latent2_dists': latent2_dists}

    def sample_prior_eval_init(self, batch_size):
        latent1_dist = self.latent1_init_prior(torch.rand(batch_size, 1))
        latent1_sample = latent1_dist.rsample()

        latent2_dist = self.latent2_init_prior(latent1_sample)
        latent2_sample = latent2_dist.rsample()

        return {'latent1_sample:': latent1_sample,
                'latent2_sample:': latent2_sample,
                'latent1_dist': latent1_dist,
                'latent2_dist': latent2_dist}

    def sample_prior_eval(self,
                          action,
                          feature_gt,
                          latent1_sample_before,
                          latent2_sample_before):
        """
        Args:
            action                 : (N, action_shape[0]) - Tensor
            feature_gt             : (N, *feature_shape) - Tensor
                                     (ground-truth-feature from the environment
                                     -> Auto-Regressive)
            latent1_sample_before  : (N, latent1_dim) - Tensor
            latent2_sample_before  : (N, latent2_dim) - Tensor
        """

        latent1_dist = self.latent1_prior(
            [latent2_sample_before, action, feature_gt])
        latent1_sample = latent1_dist.rsample()

        latent2_dist = self.latent2_prior(
            [latent1_sample_before, latent2_sample_before, action])
        latent2_sample = latent2_dist.rsample()

        return {'latent1_sample': latent1_sample,
                'latent2_sample': latent2_sample,
                'latent1_dist': latent1_dist,
                'latent2_dist': latent2_dist}

    def sample_posterior(self, features_seq, actions_seq):
        """
        Sample from posterior dynamics.
        Args:
            features_seq      : (N, S+1, 256) tensor of feature sequenses.
            actions_seq       : (N, S, *action_space) tensor of action sequenses.
        Returns:
            latent1_samples   : (N, S+1, latent1_dim)
            latent2_samples   : (N, S+1, latent2_dim)
            latent1_dists     : (S+1) list of (N, latent1_dim) distributions
            latent2_dists     : (S+1) list of (N, latent2_dim) distributions
        """
        num_sequences = actions_seq.size(1)
        features_seq = torch.transpose(features_seq, 0, 1)
        actions_seq = torch.transpose(actions_seq, 0, 1)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(num_sequences + 1):
            if t==0:
                latent1_dist = self.latent1_init_posterior(features_seq[t])
                latent1_sample = latent1_dist.rsample()

                latent2_dist = self.latent1_init_posterior(latent1_sample)
                latent2_sample = latent2_dist.rsample()

            else:
                latent1_dist = self.latent1_posterior(
                    [features_seq[t], latent2_samples[t-1], actions_seq[t-1]])
                latent1_sample = latent1_dist.rsample()

                latent2_dist = self.latent2_posterior(
                    [latent1_sample, latent2_samples[t-1], actions_seq[t-1]])
                latent2_sample = latent2_dist.rsample()

            latent1_samples.append(latent1_sample)
            latent1_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        return {'latent1_samples': latent1_samples,
                'latent2_samples': latent2_samples,
                'latent1_dists': latent1_dists,
                'latent2_dists': latent2_dists}
