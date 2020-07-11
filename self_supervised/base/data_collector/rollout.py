import gym
import numpy as np

from rlkit.samplers.rollout_functions import rollout
import rlkit.torch.pytorch_util as ptu

import self_supervised.utils.typed_dicts as td
from self_supervised.policy.skill_policy import SkillTanhGaussianPolicy
from self_supervised.utils.wrapper import SkillTanhGaussianPolicyRlkitBehaviour


class Rollouter(object):

    def __init__(self,
                 env: gym.Env,
                 policy: SkillTanhGaussianPolicy):
        self._env = env
        self._real_policy = policy
        self._rlkit_policy = SkillTanhGaussianPolicyRlkitBehaviour(policy)

    def do_rollout(self,
                max_path_length: int=None,
                render: bool=None,
                render_kwargs: dict=None) -> TransitionMapping:

        path = rollout(
            env=self._env,
            agent=self._policy,
            max_path_length=max_path_length,
            render=render,
            render_kwargs=render_kwargs
        )

        path = self._reshape_path(path)

        return TransitionMapping(**path)

    @staticmethod
    def _reshape_path(path):
        assert len(path['rewards'].shape) == len(path['terminals'].shape) == 1

        path['rewards'] = np.expand_dims(path['rewards'], 1)
        path['terminals'] = np.expand_dims(path['terminals'], 1)

        return path
