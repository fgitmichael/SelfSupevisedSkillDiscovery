import torch
import numpy as np


from self_supervised.policy.skill_policy import SkillTanhGaussianPolicy, MakeDeterministic

import self_supervised.utils.typed_dicts as td

import rlkit.torch.pytorch_util as ptu


class SkillTanhGaussianPolicyRevised(SkillTanhGaussianPolicy):

    def _check_skill(self,
                     skill: torch.Tensor):
        super()._check_skill(skill)
        assert len(skill.shape) == 1

    def get_action(self,
                   obs_np: np.ndarray,
                   skill: torch.Tensor = None,
                   deterministic: bool = False) ->np.ndarray:
        if skill is not None:
            raise NotImplementedError('This function should only be used '
                                      'with the skill already set')

        assert len(obs_np.shape) == 1
        data_dim = -1

        obs_tensor = ptu.from_numpy(obs_np)
        obs_skill_cat = torch.cat([obs_tensor, self.skill], data_dim)

        action = self.get_skill_actions(
            obs_skill_cat=obs_skill_cat,
            deterministic=deterministic
        )

        assert action.shape[-1] == self._dimensions['action_dim']
        assert len(action.shape) == 1

        return action

    @property
    def skill(self):
        return self._skill

    @skill.setter
    def skill(self, skill: torch.Tensor):
        self._check_skill(skill)
        self._skill = skill

    def set_skill(self,
                  skill: torch.Tensor):
        raise NotImplementedError('Use the property skill instead!')


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
        raise NotImplementedError('Use the property skill instead!')
