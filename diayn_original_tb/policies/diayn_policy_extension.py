from rlkit.torch.sac.diayn.policies import SkillTanhGaussianPolicy


class SkillTanhGaussianPolicyExtension(SkillTanhGaussianPolicy):

    def set_skill(self, skill: int):
        assert skill < self.skill_dim
        self.skill = skill

    def forward(
            self,
            obs,
            skill_vec=None,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        # Dims checking
        if skill_vec is None:
            assert obs.shape[-1] == self.input_size
        else:
            assert (obs.shape[-1] + skill_vec.shape[-1]) == self.input_size

        policy_return_tuple = super().forward(
            obs=obs,
            skill_vec=skill_vec,
            reparameterize=reparameterize,
            deterministic=deterministic,
            return_log_prob=return_log_prob
        )

        return policy_return_tuple


class MakeDeterministicExtension(Policy):

    def __init__(self,
                 stochastic_policy: SkillTanhGaussianPolicyExtension):
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
