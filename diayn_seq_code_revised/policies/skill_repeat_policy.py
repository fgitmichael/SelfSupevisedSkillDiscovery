import torch

from diayn_seq_code_revised.policies.skill_policy import \
    SkillTanhGaussianPolicyRevised


class SkillRepeatTanhGaussianPolicy(SkillTanhGaussianPolicyRevised):

    def __init__(self,
                 *args,
                 skill_dim: int,
                 num_skill_repeat: int,
                 **kwargs):
        self.num_skill_repeat = num_skill_repeat
        self.real_skill_dim = skill_dim
        skill_dim_repeat = self.real_skill_dim * self.num_skill_repeat

        super().__init__(
            *args,
            skill_dim=skill_dim_repeat,
            **kwargs
        )

    def get_skill_repeat(self, skill: torch.Tensor):
        assert skill.size(-1) == self.real_skill_dim
        data_dim = -1
        return torch.cat([skill] * self.num_skill_repeat, dim=data_dim)

    @property
    def skill(self):
        return self._skill

    @skill.setter
    def skill(self, skill: torch.Tensor):
        skill_rep = self.get_skill_repeat(skill)
        self._check_skill(skill_rep)
        self._skill = skill_rep

    def forward(self,
                *args,
                skill_vec=None,
                **kwargs):
        if skill_vec is not None:
            skill_vec = self.get_skill_repeat(skill_vec)

        return super().forward(
            *args,
            skill_vec=skill_vec,
            **kwargs
        )
