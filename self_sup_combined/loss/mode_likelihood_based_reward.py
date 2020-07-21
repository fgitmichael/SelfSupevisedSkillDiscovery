import torch
import torch.nn.functional as F
from self_sup_combined.network.mode_encoder import ModeEncoderSelfSupComb
import abc

from self_supervised.policy.skill_policy import SkillTanhGaussianPolicy
from self_supervised.utils.typed_dicts import ForwardReturnMapping


class RewardCalculatorBase(object, metaclass=abc.ABCMeta):

    def __init__(self,
                 skill_policy: SkillTanhGaussianPolicy,
                 mode_encoder: ModeEncoderSelfSupComb):
        self.policy = skill_policy
        self.mode_encoder = mode_encoder

        self.batch_dim = 0
        self.seq_dim = -2
        self.data_dim = -1

        self.batch_size = None
        self.seq_len = None

    @abc.abstractmethod
    def _calc_rewards(self,
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
        """
        raise NotImplementedError('To be implemented in subclass')

    def calc_rewards(self,
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
        """
        self._check_inputs(
            obs_seq=obs_seq,
            action_seq=action_seq,
            skill_gt=skill_gt
        )

        self.batch_size = obs_seq.size(self.batch_dim)
        self.seq_len = obs_seq.size(self.seq_dim)

        rewards = self._calc_rewards(
            obs_seq=obs_seq,
            action_seq=action_seq,
            skill_gt=skill_gt
        )

        self._check_rewards(rewards)

        return rewards

    def _check_inputs(self,
                      obs_seq: torch.Tensor,
                      action_seq: torch.Tensor,
                      skill_gt: torch.Tensor):
        assert len(obs_seq.shape) \
               == len(action_seq.shape) \
               == len(skill_gt.shape) == 3
        assert obs_seq.size(self.batch_dim) \
               == skill_gt.size(self.batch_dim) \
               == action_seq.size(self.batch_dim)
        assert skill_gt.size(-1) == self.policy.skill_dim
        assert action_seq.size(self.data_dim) == self.policy.action_dim
        assert obs_seq.size(self.data_dim) == self.policy.obs_dim
        assert torch.all(
            torch.stack([skill_gt[:, 0, :]] * obs_seq.size(self.seq_dim),
                        dim=self.seq_dim) \
            == skill_gt
        )

    def _check_rewards(self,
                       rewards: torch.Tensor):
        assert rewards.shape == torch.Size((self.batch_size, self.seq_len, 1))


class ReconstructionLikelyhoodBasedRewards(RewardCalculatorBase):

    @torch.no_grad()
    def _calc_rewards(self,
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
        mode_enc = self.mode_encoder(obs_seq=obs_seq.detach())
        mode_post = mode_enc['post']['dist'].loc
        assert mode_post.shape == skill_gt[:, 0, :].shape
        mode_post_repeated = torch.cat([mode_post] * self.seq_len, dim=0)
        assert obs_seq.view(self.batch_size * self.seq_len, obs_seq.size(self.data_dim))\
                   .size(self.batch_dim) \
            == mode_post_repeated.size(self.batch_dim)

        action_recon_mapping = self.policy(
            obs=obs_seq.detach().view(
                self.batch_size * self.seq_len, obs_seq.size(self.data_dim)),
            skill_vec=mode_post_repeated,
            reparameterize=False
        )
        action_recon = action_recon_mapping.action
        action_recon = action_recon\
            .view(self.batch_size, self.seq_len, action_seq.size(self.data_dim))
        assert action_recon.shape == action_seq.shape

        recon_error = torch.sum((action_seq - action_recon)**2,
                                dim=self.data_dim,
                                keepdim=True)

        return -recon_error


class ActionDiffBasedRewards(RewardCalculatorBase):

    @torch.no_grad()
    def _calc_rewards(self,
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
        """
        mode_enc = self.mode_encoder(obs_seq=obs_seq)
        mode_post = mode_enc['post']['dist'].loc
        assert mode_post.shape == skill_gt[:, 0, :].shape

        mode_post_repeated = torch.stack([mode_post] * self.seq_len, dim=self.seq_dim)
        assert mode_post_repeated.shape[:-1] == obs_seq.shape[:-1]

        actions_real_skill_mapping = self.policy(
            obs=obs_seq,
            skill_vec=skill_gt,
            reparameterize=False
        )
        action_real_skill = actions_real_skill_mapping.action

        actions_classified_skill_mapping = self.policy(
            obs=obs_seq,
            skill_vec=mode_post_repeated,
            reparameterize=False
        )
        action_classified_skill = actions_classified_skill_mapping.action

        assert action_real_skill.shape == action_classified_skill.shape

        error = torch.sum(
            (action_real_skill - action_classified_skill)**2,
            dim=self.data_dim,
            keepdim=True
        )

        return -error
