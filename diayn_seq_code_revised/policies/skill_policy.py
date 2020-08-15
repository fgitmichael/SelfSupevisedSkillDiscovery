import torch
import numpy as np
from typing import Tuple


from self_supervised.policy.skill_policy import SkillTanhGaussianPolicy, MakeDeterministic

import self_supervised.utils.typed_dicts as td

import rlkit.torch.pytorch_util as ptu


class SkillTanhGaussianPolicyRevised(SkillTanhGaussianPolicy):

    def get_action(self,
                   obs_np: np.ndarray,
                   skill: torch.Tensor = None,
                   deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if skill is not None:
            raise NotImplementedError('This function should only be used '
                                      'with the skill already set')

        assert len(obs_np.shape) == 1
        data_dim = -1

        obs_tensor = ptu.from_numpy(obs_np)
        obs_skill_cat = torch.cat([obs_tensor, self.skill], dim=data_dim)

        action = self.get_skill_actions(
            obs_skill_cat=obs_skill_cat,
            deterministic=deterministic
        )

        assert action.shape[-1] == self._dimensions['action']
        assert len(action.shape) == 1

        return action, ptu.get_numpy(self.skill)

    @property
    def skill(self):
        return self._skill

    @skill.setter
    def skill(self, skill: torch.Tensor):
        self._check_skill(skill)
        self._skill = skill

    def set_skill(self,
                  skill: torch.Tensor):
        self.skill = skill

    def forward(self,
                obs: torch.Tensor,
                skill_vec=None,
                reparameterize=True,
                return_log_prob=False,
                deterministic=False):
        # Dims checking
        if skill_vec is None:
            assert obs.shape[-1] == self.input_size
        else:
            assert (obs.shape[-1] + skill_vec.shape[-1]) == self.input_size

        if skill_vec is None:
            obs_base_call= obs[..., :self.obs_dim]
            skill_vec_base_call = obs[..., self.obs_dim:]
        else:
            obs_base_call = obs
            skill_vec_base_call = skill_vec

        for_ret_mapping = super().forward(
            obs=obs_base_call,
            skill_vec=skill_vec_base_call,
            reparameterize=reparameterize,
            return_log_prob=return_log_prob,
            deterministic=deterministic
        )

        return (
            for_ret_mapping.action,
            for_ret_mapping.mean,
            for_ret_mapping.log_std,
            for_ret_mapping.log_prob,
            for_ret_mapping.entropy,
            for_ret_mapping.std,
            for_ret_mapping.mean_action_log_prob,
            for_ret_mapping.pre_tanh_value
        )


class MakeDeterministicRevised(MakeDeterministic):

    def __int__(self,
                stochastic_policy: SkillTanhGaussianPolicyRevised
                ):
        self.stochastic_policy = stochastic_policy

    @property
    def skill_dim(self):
        raise NotImplementedError

    @property
    def skill(self):
        return self.stochastic_policy.skill

    @skill.setter
    def skill(self, skill):
        self.stochastic_policy.skill = skill

    def set_skill(self, skill):
        self.skill = skill

    @property
    def obs_dim(self):
        return self.stochastic_policy.obs_dim

    @property
    def action_dim(self):
        return self.stochastic_policy.action_dim
