import numpy as np

from diayn_cont.policy.skill_policy_obs_dim_select import SkillTanhGaussianPolicyObsSelect

from rlkit.policies.base import Policy


class SkillTanhGaussianPolicyObsSelectDIAYN(SkillTanhGaussianPolicyObsSelect):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skill = 0

    @property
    def skill(self):
        return self._skill

    @skill.setter
    def skill(self, skill_to_set):
        assert isinstance(skill_to_set, int)
        assert skill_to_set < self.skill_dim
        self._skill = skill_to_set


class MakeDeterministic(Policy):

    def __init__(self,
                 stochastic_policy: SkillTanhGaussianPolicyObsSelectDIAYN):
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
