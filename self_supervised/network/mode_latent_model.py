import torch
from typing import Union, Tuple

from mode_disent_no_ssm.network.mode_model import ModeLatentNetwork

from mode_disent_no_ssm.utils.empty_network import Empty


class ModeLatentNetworkWithEncoder(ModeLatentNetwork):

    def __init__(self,
                 obs_dim,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        if obs_dim == kwargs['feature_dim']:
            self.obs_encoder = Empty().to(self.device)
        else:
            self.obs_encoder = torch.nn.Linear(
                obs_dim, kwargs['feature_dim']).to(self.device)

        self.obs_dim = obs_dim

    def sample_mode_posterior(
            self,
            obs_seq: torch.Tensor,
            return_features: bool = False) \
            -> Union[dict, Tuple[dict, torch.Tensor]]:
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

        if return_features:
            return post, features_seq
        else:
            return post

    def reconstruct_action(self,
                           obs_seq: torch.Tensor):
        """
        Args:
            obs_seq          : (N, obs_dim, S) tensor
        """
        post, features_seq = self.sample_mode_posterior(obs_seq, return_features=True)

        action_seq_recon = self.action_decoder(
            state_rep_seq=features_seq,
            mode_sample=post['samples']
        )

        return action_seq_recon
