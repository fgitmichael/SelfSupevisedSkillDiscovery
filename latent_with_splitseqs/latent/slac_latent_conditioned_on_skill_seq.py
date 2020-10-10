import torch

from code_slac.network.base import BaseNetwork
from code_slac.network.latent import ConstantGaussian

from diayn_seq_code_revised.networks.my_gaussian import MyGaussian as Gaussian

from latent_with_splitseqs.base.latent_base import StochasticLatentNetBase


class SlacLatentNetConditionedOnSkillSeq(StochasticLatentNetBase):

    def __init__(self,
                 *args,
                 obs_dim,
                 skill_dim,
                 latent1_dim=32,
                 latent2_dim=256,
                 hidden_units=(256, 256),
                 leaky_slope=0.2,
                 dropout=0.,
                 **kwargs,
                 ):
        super(SlacLatentNetConditionedOnSkillSeq, self).__init__(*args, **kwargs)
        # We use the observations as actions for this model
        # and the infered skill as observaton

        # p(z1(0)) = N(0, I)
        self.latent1_init_prior = ConstantGaussian(latent1_dim)
        # p(z2(0) | z1(0))
        self.latent2_init_prior = Gaussian(
            input_dim=latent1_dim,
            output_dim=latent2_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope,
            dropout=dropout,
        )
        # p(z1(t+1) | z2(t), a(t))
        self.latent1_prior = Gaussian(
            input_dim=latent2_dim + obs_dim,
            output_dim=latent1_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope,
            dropout=dropout,
        )
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_prior = Gaussian(
            input_dim=latent1_dim + latent2_dim + obs_dim,
            output_dim=latent2_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope,
            dropout=dropout,
        )

        # q(z1(0) | feat(0))
        self.latent1_init_posterior = Gaussian(
            input_dim=skill_dim,
            output_dim=latent1_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope,
            dropout=dropout,
        )
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.latent2_init_posterior = self.latent2_init_prior
        # q(z1(t+1) | z2(t), a(t))
        self.latent1_posterior = Gaussian(
            input_dim=latent2_dim + obs_dim + skill_dim,
            output_dim=latent1_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope,
            dropout=dropout,
        )
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_posterior = self.latent2_prior

        self.latent1_dim = latent1_dim
        self.latent2_dim = latent2_dim

    def sample_prior(self,
                     obs_seq,
                     ):
        """
        Sample from prior feature encoding

        Args:
            obs_seq             : (N, S, obs_dim) tensor of observations
        Returns:
            latent1_samples     : (N, S+1, L1) tensor of latent samples
            latent2_samples     : (N, S+1, L2) tensor of latent samples
            latent1_dists       : (S+1) list of (N, L1) distributions
            latent2_dists       : (S+1) list of (N, L2) distributions
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        seq_len = obs_seq.size(seq_dim)
        obs_seq_seqdim_first = torch.transpose(obs_seq, batch_dim, seq_dim)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(seq_len + 1):
            if t == 0:
                # p(z1(0)) = N(0, I)
                latent1_dist = self.latent1_init_prior(obs_seq_seqdim_first[t])
                latent1_sample = latent1_dist.rsample()

                # p(z2(0) | z1(0))
                latent2_dist = self.latent2_init_prior(latent1_sample)
                latent2_sample = latent2_dist.rsample()

            else:
                # p(z1(t) | z2(t-1), a(t-1))
                latent1_dist = self.latent1_prior(
                    [latent2_samples[t-1],
                     obs_seq_seqdim_first[t-1]]
                )
                latent1_sample = latent1_dist.rsample()

                # p(z2(t) | z1(t), z2(t-1), a(t-1))
                latent2_dist = self.latent2_prior(
                    [latent1_sample,
                     latent2_samples[t-1],
                     obs_seq_seqdim_first[t-1]]
                )
                latent2_sample = latent2_dist.rsample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        latent1_samples_stacked = torch.stack(latent1_samples, dim=seq_dim)
        latent2_samples_stacked = torch.stack(latent2_samples, dim=seq_dim)

        return {
            'latent1_samples': latent1_samples_stacked,
            'latent2_samples': latent2_samples_stacked,
            'latent1_dists': latent1_dists,
            'latent2_dists': latent2_dists,
        }

    def sample_posterior(self, skill, obs_seq):
        """
        Sample from posterior

        Args:
            skill               : (N, skill_dim) tensors
            obs_seq             : (N, S, obs_dim) tensors of observations
        Returns:
            latent1_samples     : (N, S+1, L1) tensor of sampled latent vectors
            latent2_samples     : (N, S+1, L2) tensor of sampled latent vectors
            latent1_dists       : (S+1) length list of (N, L1) distributions
            latent2_dists       : (S+1) length list of (N, L2) distributions
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        seq_len = obs_seq.size(seq_dim)

        obs_seq_seqdim_first = torch.transpose(obs_seq, batch_dim, seq_dim)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(seq_len + 1):
            if t == 0:
                # q(z1(0))
                # sample from normal dist
                # (observation is only used for batchsize and device)
                latent1_dist = self.latent1_init_posterior(skill)
                latent1_sample = latent1_dist.rsample()

                # q(z2(0) | z1(0))
                latent2_dist = self.latent2_init_posterior(latent1_sample)
                latent2_sample = latent2_dist.rsample()

            else:
                # q(z1(t) | z2(t-1), obs(t-1))
                latent1_dist = self.latent1_posterior(
                    [skill,
                     latent2_samples[t-1],
                     obs_seq_seqdim_first[t-1]]
                )
                latent1_sample = latent1_dist.rsample()

                # q(z2(t) | z1(t), z2(t-1), obs(t-1))
                latent2_dist = self.latent2_posterior(
                    [latent1_sample,
                     latent2_samples[t-1],
                     obs_seq_seqdim_first[t-1]]
                )
                latent2_sample = latent2_dist.rsample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        latent1_samples_stacked = torch.stack(latent1_samples, dim=seq_dim)
        latent2_samples_stacked = torch.stack(latent2_samples, dim=seq_dim)

        return {
            'latent1_samples': latent1_samples_stacked,
            'latent2_samples': latent2_samples_stacked,
            'latent1_dists': latent1_dists,
            'latent2_dists': latent2_dists,
        }
