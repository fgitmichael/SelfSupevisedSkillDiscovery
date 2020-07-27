import numpy as np
import torch
import random


from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.core import eval_np
from rlkit.policies.base import Policy


class SkillTanhGaussianPolicyNoOHTwoDim(TanhGaussianPolicy):

    def __init__(self,
                 *args,
                 skill_dim,
                 get_skills,
                 **kwargs):
        super().__init__(*args, **kwargs)
#        assert skill_dim == 2
        self.skill_dim = skill_dim
        self.skills = get_skills()
        self.num_skills = self.skills.shape[0]

        self.skill_id = random.randint(0, self.num_skills - 1)
        self.skill = self.skills[self.skill_id]

    @property
    def skill(self):
        return self.__skill

    @skill.setter
    def skill(self, skill):
        assert isinstance(skill, np.ndarray)
        assert skill.shape == (self.skill_dim,)
        assert skill in self.skills

        self.skill_id = self.get_skill_id_from_skill(skill)[0]
        self.__skill = skill

    def set_skill(self, skill_id: int):
        assert isinstance(skill_id, int)
        self.skill = self.skills[skill_id]

    def get_skill_id_from_skill(self, skill: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(self.skills - skill, axis=1)
        idx = np.nonzero(dist < 0.0001)[0]
        return idx

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

class MakeDeterministicExtensionNoOH(Policy):

    def __init__(self,
                 stochastic_policy: SkillTanhGaussianPolicyNoOHTwoDim):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation: np.ndarray):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def set_skill(self, skill: int):
        self.stochastic_policy.set_skill(skill)

    @property
    def skill_dim(self):
        return self.stochastic_policy.skill_dim

    @property
    def skill(self):
        return self.stochastic_policy.skill

    @property
    def skill_id(self):
        return self.stochastic_policy.skill_id

    @property
    def num_skills(self):
        return self.stochastic_policy.num_skills

    def get_skill_id_from_skill(self, skill: np.ndarray) -> np.ndarray:
        return self.stochastic_policy.get_skill_id_from_skill(skill)