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
            skill_gt           : (N, S, mode_dim) tensor
        Return:
            rewards            : (N, S, 1) tensor
        Reward agent for finding discriminable skills. Reward is based on how likely is it,
        that the agent was actually command to execute the skill. In the easiest form
        this would be p(post_skill | skill_gt). To get a measure for every transition
        """
        batch_dim = 0
        seq_dim = -2
        data_dim = -1

        batch_size = obs_seq.size(batch_dim)
        seq_len = obs_seq.size(seq_dim)

        assert len(obs_seq.shape) \
               == len(action_seq.shape) \
               == len(skill_gt.shape) == 3
        assert obs_seq.size(batch_dim) \
               == skill_gt.size(batch_dim) \
               == action_seq.size(batch_dim)
        assert skill_gt.size(-1) == self.policy.skill_dim
        assert action_seq.size(data_dim) == self.policy.action_dim
        assert obs_seq.size(data_dim) == self.policy.obs_dim
        assert torch.all(
            torch.stack([skill_gt[:, 0, :]] * obs_seq.size(seq_dim), dim=seq_dim) \
            == skill_gt
        )

        mode_enc = self.mode_encoder(obs_seq=obs_seq.detach())
        mode_post = mode_enc['post']['dist'].loc
        assert mode_post.shape == skill_gt[:, 0, :].shape
        mode_post_repeated = torch.cat([mode_post] * seq_len, dim=0)
        assert obs_seq.view(batch_size * seq_len, obs_seq.size(data_dim)).size(batch_dim) \
            == mode_post_repeated.size(batch_dim)

        action_recon_mapping = self.policy(
            obs=obs_seq.detach().view(batch_size * seq_len, obs_seq.size(data_dim)),
            skill_vec=mode_post_repeated,
            reparameterize=False
        )
        action_recon = action_recon_mapping.action
        action_recon = action_recon.view(batch_size, seq_len, action_seq.size(data_dim))
        assert action_recon.shape == action_seq.shape

        recon_error = torch.sum((action_seq - action_seq)**2,
                                dim=data_dim,
                                keepdim=True)
        assert recon_error.shape == torch.Size(
            (obs_seq.size(batch_dim), obs_seq.size(seq_dim))
        )

        return -recon_error
