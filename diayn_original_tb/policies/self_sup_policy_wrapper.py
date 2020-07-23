import torch
import random
import numpy as np


from self_supervised.policy.skill_policy import SkillTanhGaussianPolicy
import self_supervised.utils.typed_dicts as td

import rlkit.torch.pytorch_util as ptu

class RlkitWrapperForMySkillPolicy(SkillTanhGaussianPolicy):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.skill = 0

    def get_action(self,
                   obs_np: np.ndarray,
                   skill: torch.Tensor = None,
                   deterministic: bool = False):
        # Ignore skill
        if skill is not None:
            raise NotImplementedError('This wrapper simulates the rlkit behaviour.'
                                      'In rkit no skill is given as arg.')

        skill_vec = ptu.zeros(self.skill_dim)
        skill_vec[self.skill] += 1

        action_mapping = super().get_action(
            obs_np=obs_np,
            skill=skill_vec,
            deterministic=deterministic,
        )

        # Rlkit wants oh
        skill_oh = np.zeros((self.skill_dim))
        skill_oh[action_mapping.agent_info['skill']] += 1

        return action_mapping.action, {"skill": skill_oh}

    def skill_reset(self):
        self.skill = random.randint(0, self.skill_dim - 1)

    def set_skill(self,
                  skill: int):
        assert skill < self.skill_dim
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
