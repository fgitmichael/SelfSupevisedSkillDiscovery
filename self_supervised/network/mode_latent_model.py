import torch
from typing import Union, Tuple

from mode_disent_no_ssm.network.mode_model import ModeLatentNetwork

from mode_disent_no_ssm.utils.empty_network import Empty


class ModeLatentNetworkWithEncoder(ModeLatentNetwork):

    def __init__(self,
                 obs_dim,
                 **kwargs,
                 ):
        if kwargs['feature_dim'] is None:
            kwargs['feature_dim'] = obs_dim

        super().__init__(**kwargs)

        if obs_dim == kwargs['feature_dim']:
            self.obs_encoder = Empty().to(self.device)
        else:
            self.obs_encoder = torch.nn.Linear(
                obs_dim, kwargs['feature_dim']).to(self.device)

        self.obs_dim = obs_dim

    def sample_mode_posterior_with_features(
            self,
            obs_seq: torch.Tensor)\
            -> Tuple[dict, torch.Tensor]:
        """
        Args:
            obs_seq          : (N, obs_dim, S) tensor
        """
        assert obs_seq.size(1) == self.obs_dim
        assert len(obs_seq.shape) == 3

        obs_seq = obs_seq.transpose(0, 1)
        assert obs_seq.size(2) == self.obs_dim

        features_seq = self.obs_encoder(obs_seq)

        post = super().sample_mode_posterior(
            features_seq=features_seq
        )

        return post, features_seq

    def sample_mode_posterior(self,
                              obs_seq: torch.Tensor) -> dict:
        post, _ = self.sample_mode_posterior_with_features(obs_seq)
        return post

    def reconstruct_action(self,
                           features_seq,
                           mode_sample):
        """
        Args:
            features_seq        : (N, S, feature_dim) tensor
            mode_sample         : (N, mode_dim) tensor
        Return:
            action_recon_dist   : tuple of ...
                dists           : (N, S) Distribution over decoded actions
                samples         : (N, S) tensor
        """

        action_recon_dist = self.action_decoder(
            state_rep_seq=features_seq,
            mode_sample=mode_sample
        )

        return action_recon_dist
