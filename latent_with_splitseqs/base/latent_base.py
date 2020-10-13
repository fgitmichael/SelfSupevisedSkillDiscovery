import torch
import abc

from code_slac.network.base import BaseNetwork


class StochasticLatentNetBase(BaseNetwork, metaclass=abc.ABCMeta):

    def __init__(self,
                 beta_anneal):
        super(StochasticLatentNetBase, self).__init__()

        self.beta_anneal = beta_anneal
        if beta_anneal is not None:
            self._check_beta_anneal(beta_anneal)
            self.beta = self.beta_anneal['start']

    def anneal_beta(self):
        if self.beta_anneal is not None:
            if self.beta_anneal['beta'] < self.beta_anneal['end']:
                beta = self.beta + self.beta_anneal['add']
                if self.beta > self.beta_anneal['end']:
                    beta = self.beta_anneal['end']
                self.beta = beta

    @property
    def beta(self):
        if self.beta_anneal is None:
            return 1.
        else:
            return self.beta_anneal['beta']

    @beta.setter
    def beta(self, val):
        self.beta_anneal['beta'] = val

    def _check_beta_anneal(self, beta_anneal: dict):
        assert 'start' in beta_anneal.keys()
        assert 'add' in beta_anneal.keys()
        assert 'end' in beta_anneal.keys()

    @abc.abstractmethod
    def sample_prior(self, obs_seq):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_posterior(self, skill, obs_seq):
        raise NotImplementedError

    def forward(self, skill, obs_seq):
        """
        Args:
            skill                   : (N, skill_dim) tensor (skill batch)
            obs_seq                 : (N, S, obs_dim) tensor (sequence batch)
        Return:
            pri
                latent1_samples     : (N, S+1, L1) tensor of sampled latent vectors
                latent2_samples     : (N, S+1, L2) tensor of sampled latent vectors
                latent1_dists       : (S+1) length list of (N, L1) distributions
                latent2_dists       : (S+1) length list of (N, L2) distributions
            post
                latent1_samples     : (N, S+1, L1) tensor of sampled latent vectors
                latent2_samples     : (N, S+1, L2) tensor of sampled latent vectors
                latent1_dists       : (S+1) length list of (N, L1) distributions
                latent2_dists       : (S+1) length list of (N, L2) distributions
        """
        pri = self.sample_prior(
            obs_seq=obs_seq,
        )
        post = self.sample_posterior(
            obs_seq=obs_seq,
            skill=skill,
        )

        return dict(
            pri=pri,
            post=post,
        )
