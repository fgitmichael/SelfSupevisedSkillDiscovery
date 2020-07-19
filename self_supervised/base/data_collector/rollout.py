import gym
import numpy as np
import torch

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
                   skill: torch.Tensor,
                   max_path_length: int=None,
                   render: bool=None,
                   render_kwargs: dict=None) -> td.TransitionModeMapping:
        """
        Args:
            skill        : (skill_dim) tensor
        """
        self._real_policy.set_skill(skill)

        path = rollout(
            env=self._env,
            agent=self._rlkit_policy,
            max_path_length=max_path_length,
            render=render,
            render_kwargs=render_kwargs
        )

        path = self._reshape(path)

        mode_np = ptu.get_numpy(self._real_policy.skill)
        mode_np_seq = np.stack([mode_np] * max_path_length, axis=1)

        return td.TransitionModeMapping(
            obs=path['observations'],
            action=path['actions'],
            reward=path['rewards'],
            next_obs=path['next_observations'],
            terminal=path['terminals'],
            mode=mode_np_seq,
            #agent_infos=path['agent_infos'],
            #env_infos=path['env_infos'],
        )

    def _reshape(self, path: dict):
        for k, v in path.items():
            if type(path[k]) is np.ndarray:
                path[k] = np.transpose(path[k])

        return path

    def reset(self):
        self._env.reset()
