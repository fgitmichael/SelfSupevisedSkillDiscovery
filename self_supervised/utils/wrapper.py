from typing import Tuple, Dict, Union
import torch
import numpy as np

from self_supervised.policy.skill_policy import SkillTanhGaussianPolicy, MakeDeterministic


class SkillTanhGaussianPolicyRlkitBehaviour():

    def __init__(self,
                 to_be_converted: Union[
                     SkillTanhGaussianPolicy,
                     MakeDeterministic
                 ]
                 ):
        self.policy = to_be_converted

    def get_action(
            self,
            obs_np: np.ndarray,
            skill: torch.Tensor = None,
            deterministic: bool = None) -> Tuple[np.ndarray, dict]:
        action_mapping = self.policy.get_action(
                obs_np=obs_np,
                skill=skill,
                deterministic=deterministic
        )

        return action_mapping.action, action_mapping.agent_info

    def reset(self):
        return self.policy.reset()

