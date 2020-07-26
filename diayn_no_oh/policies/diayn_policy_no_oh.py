import numpy as np
import torch
import random


from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.core import eval_np
from rlkit.policies.base import Policy

from diayn_no_oh.utils.hardcoded_grid_two_dim import get_grid


class SkillTanhGaussianPolicyNoOHTwoDim(TanhGaussianPolicy):

    def __init__(self,
                 *args,
                 skill_dim,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert skill_dim == 2
        self.skill_dim = skill_dim
        self.skills = get_grid()
        self.num_skills = self.skills.shape[0]

        self.skill_id = random.randint(0, self.num_skills - 1)
        self.skill = self.skills[self.skill_id]

    @property
    def skill(self):
        return self.__skill

    @skill.setter
    def skill(self, skill):
        assert isinstance(skill, np.ndarray)
        assert skill.shape == (2,)
        assert skill in self.skills

        self.skill_id = np.where(self.skills == skill)[0][0]
        self.__skill = skill

    def set_skill(self, skill_id):
        self.skill = self.skills[skill_id]

    def get_action(self, obs_np, deterministic=False):
        assert len(obs_np.shape) == 1
        obs_skill_cat = np.concatenate((obs_np, self.skill), axis=0)
        obs_skill_cat = np.expand_dims(obs_skill_cat, axis=0)
        actions = self.get_actions(obs_skill_cat, deterministic=deterministic)

        return actions[0, :], {"skill": self.skill}

    def skill_reset(self):
        self.skill = self.skills[random.randint(0, self.num_skills - 1)]

    def forward(
            self,
            obs,
            skill_vec=None,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        assert isinstance(obs, torch.Tensor)

        # Dimension Checking
        if skill_vec is None:
            assert obs.shape[-1] == self.input_size
        else:
            assert obs.shape[-1] + skill_vec.shape[-1] == self.input_size

        if skill_vec is None:
            h = obs

        else:
            assert isinstance(skill_vec, torch.Tensor)
            h = torch.cat((obs, skill_vec), dim=1)

        policy_return_tuple = super().forward(
            obs=h,
            reparameterize=reparameterize,
            deterministic=deterministic,
            return_log_prob=return_log_prob
        )

        return policy_return_tuple
