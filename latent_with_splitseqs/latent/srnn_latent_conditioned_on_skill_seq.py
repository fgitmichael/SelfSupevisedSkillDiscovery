import torch
import torch.nn as nn
from typing import Type

from latent_with_splitseqs.base.latent_base import StochasticLatentNetBase

from code_slac.network.base import BaseNetwork

from self_supervised.network.flatten_mlp import FlattenMlp as \
    MyFlattenMlp


class SRNNLatentConditionedOnSkillSeq(BaseNetwork):

    def __init__(self,
                 obs_dim,
                 skill_dim,
                 filter_net_params,
                 deterministic_latent_net: nn.GRU,
                 stochastic_latent_net_class: Type[StochasticLatentNetBase],
                 stochastic_latent_net_class_params,
                 ):
        """
        Args:
            deterministic_latent_net                : instance of a deterministic
                                                      state space model class
                                                      such as a rnn
            stochastic_latent_net_class             : class inheritated
                                                      from StochasticLatentNetBase
            stochastic_latent_net_class_params      : dict with keys neede   d to
                                                      instantiate a instance of
                                                      stochastic_latent_net_class
        """
        super(SRNNLatentConditionedOnSkillSeq, self).__init__()

        self.det_latent = deterministic_latent_net
        self.det_latent_dim = self.det_latent.hidden_size
        assert self.det_latent.input_size == obs_dim

        self.stoch_latent = stochastic_latent_net_class(
            obs_dim=self.det_latent_dim,
            skill_dim=skill_dim,
            **stochastic_latent_net_class_params,
        )
        self.stoch_latent_dim = self.stoch_latent.latent_dim

        # a = g(d_t, skill)
        activation = nn.LeakyReLU(filter_net_params.pop('leaky_relu'))
        self.pre_filter_net = MyFlattenMlp(
            input_size=self.det_latent_dim + skill_dim,
            output_size=self.det_latent_dim + skill_dim,
            hidden_activation=activation,
            **filter_net_params,
        )

    @property
    def latent_dim(self):
        return self.det_latent_dim + self.stoch_latent_dim

    @property
    def beta(self):
        return self.stoch_latent.beta

    @beta.setter
    def beta(self, val):
        self.stoch_latent.beta = val

    def anneal_beta(self):
        self.stoch_latent.anneal_beta()

    def sample_prior(self, obs_seq):
        """
        Args:
            obs_seq             : (N, S, obs_dim) tensor
        Returns:
            latent_samples      : (N, S, det_latent_dim + stoch_latent_dim) tensor
            latent_dists        : list of S (N, det_latent_dim + stoch_latent_dim)
                                  distributions
        """
        data_dim = -1
        seq_dim = 1
        batch_dim = 0
        seq_len = obs_seq.size(seq_dim)

        det_latent_samples, _ = self.det_latent(obs_seq)

        pri_dict = self.stoch_latent.sample_prior_samples_cat(
            obs_seq=det_latent_samples
        )
        seq_idx_start = pri_dict['latent_samples'].size(seq_dim) - seq_len
        assert seq_idx_start >= 0
        stoch_latent_samples = pri_dict['latent_samples'][:, seq_idx_start:, :]
        assert stoch_latent_samples.shape[:-1] == det_latent_samples.shape[:-1]

        latent_samples = torch.cat(
            [det_latent_samples,
             stoch_latent_samples],
            dim=data_dim
        )
        latent_dists = pri_dict['latent_dists']

        return dict(
            latent_samples=latent_samples,
            latent_dists=latent_dists,
        )

    def sample_posterior(self, obs_seq, skill):
        """
        Args:
            obs_seq             : (N, S, obs_dim) tensor
            skill               : (N, skill_dim) tensor
        Returns:
            latent_samples      : (N, S, det_latent_dim + stoch_latent_dim) tensor
            latent_dists        : list of S (N, stoch_latent_dim)
                                  distributions
        """
        data_dim = -1
        seq_dim = 1
        batch_dim = 0
        seq_len = obs_seq.size(seq_dim)

        det_latent_samples, _ = self.det_latent(obs_seq)
        skill_seq = torch.stack([skill] * seq_len, dim=seq_dim)
        pre_filtered = self.pre_filter_net(
            det_latent_samples,
            skill_seq,
        )

        post_dict = self.stoch_latent.sample_posterior_samples_cat(
            obs_seq=pre_filtered[..., :self.det_latent_dim],
            skill_seq=pre_filtered[..., self.det_latent_dim:],
        )
        stoch_latent_samples = post_dict['latent_samples']

        latent_samples = torch.cat(
            [det_latent_samples,
             stoch_latent_samples],
            dim=data_dim
        )
        latent_dists = post_dict['latent_dists']

        return dict(
            latent_samples=latent_samples,
            latent_dists=latent_dists,
        )

    def forward(self, obs_seq, skill):
        pri_dict = self.sample_prior(obs_seq)
        post_dict = self.sample_posterior(
            obs_seq=obs_seq,
            skill=skill,
        )
        return dict(
            pri_dict=pri_dict,
            post_dict=post_dict,
        )
