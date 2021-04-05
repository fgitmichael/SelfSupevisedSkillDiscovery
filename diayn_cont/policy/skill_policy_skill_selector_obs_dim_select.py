import torch

from diayn_cont.policy.skill_policy_with_skill_selector \
    import SkillTanhGaussianPolicyWithSkillSelector


class SkillTanhGaussianPolicyWithSkillSelectorObsSelect(
    SkillTanhGaussianPolicyWithSkillSelector):

    def __init__(
            self,
            *args,
            obs_dim_real: int = None,
            obs_dims_selected: tuple = (),
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        # Sanity check
        assert obs_dim_real >= len(obs_dims_selected)
        if len(obs_dims_selected) != 0:
            assert obs_dim_real is not None

        self.used_obs_dims = obs_dims_selected
        self.obs_dim_real = obs_dim_real if obs_dim_real is not None else self.obs_dim

    def select_obs_dims(
            self,
            obs: torch.Tensor,
            skill_vec: torch.Tensor
    ) -> tuple:
        if skill_vec is None:
            obs_recovered = obs[..., :self.obs_dim_real]
            skill_vec_recoverd = obs[..., self.obs_dim_real:]
            obs_recovered = obs_recovered[..., self.used_obs_dims]
            obs = torch.cat((obs_recovered, skill_vec_recoverd), dim=1)

        else:
            assert obs.shape[-1] == self.obs_dim_real
            obs = obs[..., self.used_obs_dims]

        return obs, skill_vec

    def forward(
            self,
            obs,
            skill_vec=None,
            **kwargs
    ):
        obs, skill_vec = self.select_obs_dims(obs, skill_vec)
        return super().forward(
            obs=obs,
            skill_vec=skill_vec,
            **kwargs
        )
