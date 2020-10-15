import torch

from code_slac.network.latent import ConstantGaussian

from latent_with_splitseqs.base.latent_base \
    import StochasticLatentNetBase

from diayn_seq_code_revised.networks.my_gaussian import MyGaussian as Gaussian

import rlkit.torch.pytorch_util as ptu

class OneLayeredStochasticLatent(StochasticLatentNetBase):

    def __init__(self,
                 *args,
                 obs_dim,
                 skill_dim,
                 latent_dim=256,
                 hidden_units=(256, 256),
                 leaky_slope=0.2,
                 dropout=0.,
                 **kwargs):
        super(OneLayeredStochasticLatent, self).__init__(*args, **kwargs)

        self.latent_prior = Gaussian(
            input_dim=latent_dim + obs_dim,
            output_dim=latent_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope,
            dropout=dropout,
        )

        self.latent_posterior = Gaussian(
            input_dim=latent_dim + obs_dim + skill_dim,
            output_dim=latent_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope,
            dropout=dropout,
        )

        self._latent_dim = latent_dim

    @property
    def latent_dim(self):
        return self._latent_dim

    def sample_prior(self,
                     obs_seq):
        """
        Sample from prior feature encoding

        Args:
            obs_seq             : (N, S, obs_dim) tensor of observations
        Returns:
            latent_samples     : (N, S, L1) tensor of latent samples
            latent_dists       : (S) list of (N, L2) distributions
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size, seq_len, obs_dim = obs_seq.shape

        obs_seq_seqdim_first = torch.transpose(obs_seq, batch_dim, seq_dim)

        latent_samples = []
        latent_dists = []
        for t in range(seq_len):
            if t == 0:
                latent_dist = self.latent_prior(
                    [ptu.zeros(batch_size, self.latent_dim),
                     obs_seq_seqdim_first[t]]
                )
                latent_sample = latent_dist.rsample()

            else:
                latent_dist = self.latent_prior(
                    [latent_samples[t-1],
                     obs_seq_seqdim_first[t-1]]
                )
                latent_sample = latent_dist.rsample()

            latent_samples.append(latent_sample)
            latent_dists.append(latent_dist)

        latent_samples_stacked = torch.stack(latent_samples, dim=seq_dim)
        return dict(
            samples=latent_samples_stacked,
            dists=latent_dists,
        )

    def sample_posterior(self, skill, obs_seq):
        """
        Sample from prior feature encoding

        Args:
            obs_seq             : (N, S, obs_dim) tensor of observations
            skill               : (N, skill_dim) tensor
        Returns:
            latent_samples     : (N, S, L1) tensor of latent samples
            latent_dists       : (S) list of (N, L2) distributions
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size, seq_len, obs_dim = obs_seq.shape

        obs_seq_seqdim_first = torch.transpose(obs_seq, batch_dim, seq_dim)

        latent_samples = []
        latent_dists = []

        for t in range(seq_len):
            if t == 0:
                latent_dist = self.latent_posterior(
                    [ptu.zeros(batch_size, self.latent_dim),
                     obs_seq_seqdim_first[t],
                     skill]
                )
                latent_sample = latent_dist.rsample()

            else:
                latent_dist = self.latent_posterior(
                    [latent_samples[t-1],
                     obs_seq_seqdim_first[t],
                     skill]
                )
                latent_sample = latent_dist.rsample()

            latent_dists.append(latent_dist)
            latent_samples.append(latent_sample)

        latent_samples_stacked = torch.stack(latent_samples, dim=seq_dim)
        return dict(
            samples=latent_samples_stacked,
            dists=latent_dists,
        )

    def sample_prior_samples_cat(self, *args, **kwargs):
        pri_dict = self.sample_prior(*args, **kwargs)

        return dict(
            latent_samples=pri_dict['samples'],
            latent_dists=pri_dict['dists'],
        )

    def sample_posterior_samples_cat(self, *args, **kwargs):
        post_dict = self.sample_posterior(*args, **kwargs)

        return dict(
            latent_samples=post_dict['samples'],
            latent_dists=post_dict['dists'],
        )
