import gym
import numpy as np
import torch
from typing import Union

from rlkit.samplers.rollout_functions import rollout
import rlkit.torch.pytorch_util as ptu

from diayn_original_tb.policies.diayn_policy_extension import \
    SkillTanhGaussianPolicyExtension, MakeDeterministicExtension

import self_supervised.utils.typed_dicts as td


# TODO: create absrtract base class for rollouters
class Rollouter(object):

    def __init__(self,
                 env: gym.Env,
                 policy: Union[
                     SkillTanhGaussianPolicyExtension,
                     MakeDeterministicExtension]
                 ):
        self._env = env
        self._policy = policy

    def do_rollout(self,
                   skill: int,
                   max_path_length: int=None,
                   render: bool=None,
                   render_kwargs: dict=None) -> td.TransitionModeMapping:
        """
        Args:
            skill        : skill id integer (< skill_dim, will be transfered to one-hot)
        """
        assert skill < self._policy.skill_dim
        self._policy.set_skill(skill)

        path = rollout(
            env=self._env,
            agent=self._policy,
            max_path_length=max_path_length,
            render=render,
            render_kwargs=render_kwargs
        )

        path = self._reshape(path)

        # self._policy.skill is integer (id of skill), but here we want one hot
        data_dim = -2
        seq_dim = -1
        num_classes = self._policy.skill_dim
        skill = self._policy.skill
        mode_np_oh = np.eye(num_classes)[skill]
        mode_np_seq = np.stack([mode_np_oh] * max_path_length, axis=seq_dim)

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
