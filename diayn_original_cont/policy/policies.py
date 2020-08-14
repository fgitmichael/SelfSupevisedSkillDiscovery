from diayn_no_oh.policies.diayn_policy_no_oh import \
    MakeDeterministicExtensionNoOH, \
    SkillTanhGaussianPolicyNoOHTwoDim
from diayn_seq_code_revised.base.skill_selector_base import \
    SkillSelectorBase

class SkillTanhGaussianPolicyCont(SkillTanhGaussianPolicyNoOHTwoDim):

    def __init__(self,
                 *args,
                 skill_selector_cont: SkillSelectorBase,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.skill_selector_cont = skill_selector_cont

    def skill_reset(self):
        self.skill = self.skill_selector_cont.get_random_skill()

    @property
    def eval_grid(self):
        return self.skill_selector_cont.get_skill_grid()


class MakeDeterministicCont(MakeDeterministicExtensionNoOH):

    @property
    def eval_grid(self):
        return self.stochastic_policy.eval_grid()
