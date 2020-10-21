import torch
import torch.nn as nn
import torch.distributions as torch_dist

from latent_with_splitseqs.latent.slac_latent_conditioned_on_skill_seq \
    import SlacLatentNetConditionedOnSkillSeq

from diayn_seq_code_revised.networks.my_gaussian import MyGaussian as Gaussian


class SlacLatentNetConditionedOnSkillSeqSmoothingPosterior(
    SlacLatentNetConditionedOnSkillSeq):

    def __init__(self,
                 *args,
                 obs_dim,
                 skill_dim,
                 hidden_units,
                 dropout,
                 leaky_slope,
                 smoothing_rnn_hidden_size: int,
                 res_q_posterior=True,
                 **kwargs
                 ):
        super(SlacLatentNetConditionedOnSkillSeqSmoothingPosterior, self).__init__(
            *args,
            obs_dim=obs_dim,
            skill_dim=skill_dim,
            hidden_units=hidden_units,
            dropout=dropout,
            leaky_slope=leaky_slope,
            **kwargs
        )

        # Smoothing posterior rnn running backwards in time
        # q(h(t) | s(t), skill(t))
        self.smoothing_rnn = nn.GRU(
            input_size=obs_dim + skill_dim,
            hidden_size=smoothing_rnn_hidden_size,
            batch_first=True,
        )

        # q(z1(t+1) | z2(t), h(t+1))
        self.latent1_posterior = Gaussian(
            input_dim=self.latent2_dim + smoothing_rnn_hidden_size,
            output_dim=self.latent1_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope,
            dropout=dropout,
        )

        self.res_q_posterior = res_q_posterior

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

        # Run smoothing rnn backward in time
        skill_seq = torch.stack(
            [skill] * obs_seq.size(seq_dim),
            dim=seq_dim
        )
        obs_skill_seq = torch.cat(
            [obs_seq, skill_seq],
            dim=data_dim
        )
        reverse_idx = torch.arange(seq_len - 1, -1, -1)
        rnn_input_seq = obs_skill_seq[:, reverse_idx, :]
        hidden_rnn_seq, _ = self.smoothing_rnn(rnn_input_seq)

        # Sample posterior
        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []
        hidden_rnn_seqdim_first = torch.transpose(hidden_rnn_seq, batch_dim, seq_dim)
        obs_seq_seqdim_first = torch.transpose(obs_seq, batch_dim, seq_dim)

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
                if self.res_q_posterior:
                    # q(z1(t) | z2(t-1), h(t))
                    latent1_dist_residual = self.latent1_posterior(
                        [latent2_samples[t-1],
                         hidden_rnn_seqdim_first[t-1]]
                    )
                    with torch.no_grad():
                        latent1_dist_prior = self.latent1_prior(
                            [latent2_samples[t-1],
                             obs_seq_seqdim_first[t-1]]
                        )
                    latent1_dist = torch_dist.Normal(
                        loc=latent1_dist_residual.loc + latent1_dist_prior.loc.detach(),
                        scale=latent1_dist_residual.scale,
                    )
                    latent1_sample = latent1_dist.rsample()

                else:
                    # q(z1(t) | z2(t-1), h(t))
                    latent1_dist = self.latent1_posterior(
                        [latent2_samples[t - 1],
                         hidden_rnn_seqdim_first[t - 1]]
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
