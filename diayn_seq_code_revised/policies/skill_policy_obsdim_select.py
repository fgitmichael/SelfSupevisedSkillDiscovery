import torch
import numpy as np

from diayn_seq_code_revised.policies.skill_policy import \
    SkillTanhGaussianPolicyRevised

import rlkit.torch.pytorch_util  as ptu


class SkillTanhGaussianPolicyRevisedObsSelect(
    SkillTanhGaussianPolicyRevised):

    def __init__(self,
                 *args,
                 obs_dim_real: int = None,
                 obs_dims_selected: tuple = (),
                 **kwargs):
        super(SkillTanhGaussianPolicyRevised, self).__init__(
            *args,
            **kwargs
        )

        # Sanity check
        if len(obs_dims_selected) != 0:
            assert obs_dim_real is not None

        self.used_obs_dims = obs_dims_selected
        self.obs_dim_real = obs_dim_real if obs_dim_real is not None else self.obs_dim

    def get_skill_actions(self,
                          obs_skill_cat: torch.Tensor,
                          deterministic: bool=False
                          ) -> np.ndarray:
        action_tensor = self.forward(
            obs=obs_skill_cat,
        )[0]

        return ptu.get_numpy(action_tensor)

    def recover_obs_skillvec(self,
                  obs: torch.Tensor,
                  skill_vec: torch.Tensor = None,
                  ):
        # Dim Check
        if skill_vec is None:
            assert obs.shape[-1] == self.obs_dim_real + self.skill_dim
        else:
            assert (obs.shape[-1] + skill_vec.shape[-1]) \
                   == self.obs_dim_real + self.skill_dim

        if skill_vec is None:
            obs_base_call_normal = obs[..., :self.obs_dim_real]
            skill_vec_base_call = obs[..., self.obs_dim_real:]
        else:
            obs_base_call_normal = obs
            skill_vec_base_call = skill_vec

        # Accunt for obs dimensions selected
        if self.used_obs_dims is not None:
            obs_base_call = obs_base_call_normal[..., self.used_obs_dims]
        else:
            obs_base_call = obs_base_call_normal

        return obs_base_call, skill_vec_base_call
