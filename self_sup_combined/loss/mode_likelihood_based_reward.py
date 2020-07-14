import torch
import torch.nn.functional as F
from self_sup_combined.network.mode_encoder import ModeEncoderSelfSupComb

from self_supervised.policy.skill_policy import SkillTanhGaussianPolicy
from self_supervised.utils.typed_dicts import ForwardReturnMapping


class ReconstructionLikelyhoodBasedRewards():

    def __init__(self,
                 skill_policy: SkillTanhGaussianPolicy,
                 mode_encoder: ModeEncoderSelfSupComb):
        self.policy = skill_policy
        self.mode_encoder = mode_encoder

    @torch.no_grad()
    def mode_likely_based_rewards(self,
                                  obs_seq: torch.Tensor,
                                  action_seq: torch.Tensor,
                                  skill_gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_seq            : (N, S, obs_dim)
            action_seq         : (N, S, action_dim)
            skill_gt           : (N, mode_dim) tensor
        Reward agent for finding discriminable skills. Reward is based on how likely is it,
        that the agent was actually command to execute the skill. In the easiest form
        this would be p(post_skill | skill_gt). To get a measure for every transition
        """
        batch_dim = 0
        seq_dim = -2
        data_dim = -1

        assert len(obs_seq.shape) == len(action_seq.shape) == 3
        assert len(skill_gt.shape) == 2
        assert obs_seq.size(batch_dim) \
               == skill_gt.size(batch_dim) \
               == action_seq.size(batch_dim)
        assert skill_gt.size(-1) == self.policy.skill_dim
        assert action_seq.size(data_dim) == self.policy.dimensions['action_dim']
        assert obs_seq.size(data_dim) == self.policy.dimensions['obs_dim']

        mode_enc = self.mode_encoder(obs_seq=obs_seq.detach())
        mode_post = mode_enc['post'].loc
        assert mode_post.shape == skill_gt.shape

        action_recon_mapping = self.policy(
            obs=obs_seq.detach(),
            skill_vec=mode_post,
            reparameterize=False
        )
        action_recon = action_recon_mapping.action
        assert action_recon.shape == action_seq.shape

        recon_error = torch.sum((action_seq - action_seq)**2, dim=data_dim)
        assert recon_error.shape == torch.Size(
            (obs_seq.size(batch_dim), obs_seq.size(seq_dim))
        )

        return -recon_error

















