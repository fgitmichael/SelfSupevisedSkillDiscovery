from typing import Tuple, Dict
import torch
import numpy as np

from self_supervised.policy.skill_policy import SkillTanhGaussianPolicy


class SkillTanhGaussianPolicyRlkitBehaviour():

    def __init__(self, to_be_converted: SkillTanhGaussianPolicy):
        self.policy = to_be_converted

    def get_action(
            self,
            obs_np: np.ndarray,
            skill: torch.Tensor = None,
            deterministic: bool = False) -> Tuple[np.ndarray, dict]:
        action_mapping = self.policy.get_action(
            obs_np=obs_np,
            skill=skill,
            deterministic=deterministic
        )

        return action_mapping.action, action_mapping.agent_info

    def reset(self):
        return self.policy.reset()

