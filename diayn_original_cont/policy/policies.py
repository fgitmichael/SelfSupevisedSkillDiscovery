import torch

from diayn_seq_code_revised.base.skill_selector_base import \
    SkillSelectorBase
from diayn_original_tb.policies.self_sup_policy_wrapper import \
    RlkitWrapperForMySkillPolicy, \
    MakeDeterministicMyPolicyWrapper

import rlkit.torch.pytorch_util as ptu


class SkillTanhGaussianPolicyExtensionCont(RlkitWrapperForMySkillPolicy):

    def __init__(self,
                 *args,
                 skill_selector_cont: SkillSelectorBase,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.skill_selector_cont = skill_selector_cont
        self.skill = None

    def skill_reset(self):
           self.skill = self.skill_selector_cont.get_random_skill()

    @property
    def eval_grid(self):
        return self.skill_selector_cont.get_skill_grid()

    @property
    def skill(self):
        if self._skill is None:
            return ptu.randn(self.skill_dim)
        return self._skill

    @skill.setter
    def skill(self, skill):
        self._skill = skill

    def get_skill_vec_from_self_skill(self):
        return self.skill

    def get_skill_for_return(self, action_mapping):
        return action_mapping.agent_info['skill']


class MakeDeterministicCont(MakeDeterministicMyPolicyWrapper):

    @property
    def eval_grid(self):
        return self.stochastic_policy.eval_grid()

    @property
    def skill(self):
        return self.stochastic_policy.skill

    @skill.setter
    def skill(self, skill):
        self.stochastic_policy.skill = skill

    @property
    def obs_dim(self):
        return self.stochastic_policy.obs_dim

    @property
    def action_dim(self):
        return self.stochastic_policy.action_dim

    @property
    def skill_dim(self):
        return self.stochastic_policy.skill_dim

