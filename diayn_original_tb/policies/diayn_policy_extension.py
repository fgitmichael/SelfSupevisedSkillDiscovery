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

        super().forward(
            obs=obs,
            skill_vec=skill_vec,
            reparameterize=reparameterize,
            deterministic=deterministic,
            return_log_prob=return_log_prob
        )