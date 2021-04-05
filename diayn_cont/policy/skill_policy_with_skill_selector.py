import torch
import numpy as np

from rlkit.torch.sac.diayn.policies import SkillTanhGaussianPolicy
from rlkit.policies.base import Policy
import rlkit.torch.pytorch_util as ptu

from seqwise_cont_skillspace.data_collector.skill_selector_cont_skills import \
    SkillSelectorContinous


class SkillTanhGaussianPolicyWithSkillSelector(SkillTanhGaussianPolicy):

    def __init__(
            self,
            *args,
            obs_dim,
            skill_selector: SkillSelectorContinous,
            **kwargs
    ):
        super().__init__(
            *args,
            obs_dim=obs_dim,
            **kwargs
        )
        self.obs_dim = obs_dim
        self.skill_selector = skill_selector
        self._skill = skill_selector.get_random_skill()

    @property
    def skill(self):
        return self._skill

    @skill.setter
    def skill(self, skill_to_set):
        self._check_skill(skill_to_set)
        self._skill = skill_to_set

    def _check_skill(self,
                     skill: torch.Tensor):
        # In base __init__ method skill is set to zero one time
        assert isinstance(skill, torch.Tensor) or skill == 0
        if isinstance(skill, torch.Tensor):
            assert skill.shape[-1] == self.skill_dim

    def skill_reset(self):
        self.skill = self.skill_selector.get_random_skill()

    def get_action(self, obs_np, deterministic=False):
        """
        In base method skill vec is generated via one hot encodings,
        which does work with continuous skills.
        In
            obs_np          : observation (obs_dim, )
        """
        skill_vec = ptu.get_numpy(self.skill)
        obs_np = np.concatenate((obs_np, skill_vec), axis=0)
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {"skill": skill_vec}


class MakeDeterministic(Policy):

    def __init__(self,
                 stochastic_policy: SkillTanhGaussianPolicyWithSkillSelector):
        self.stochastic_policy = stochastic_policy

    def get_action(self,
                   obs_np: np.ndarray,
                   deterministic: bool = None):
        assert deterministic is None

        return self.stochastic_policy.get_action(
            obs_np,
            deterministic=True
        )

    @property
    def skill_dim(self):
        return self.stochastic_policy.skill_dim

    @property
    def skill(self):
        return self.stochastic_policy.skill

    @skill.setter
    def skill(self, skill_to_set):
        self.stochastic_policy.skill = skill_to_set

    @property
    def skill_selector(self):
        return self.stochastic_policy.skill_selector
