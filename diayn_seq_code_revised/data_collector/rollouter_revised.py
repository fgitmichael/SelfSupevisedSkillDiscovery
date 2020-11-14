import gym
from typing import Union
import numpy as np

from diayn_seq_code_revised.policies.skill_policy import \
    SkillTanhGaussianPolicyRevised, MakeDeterministicRevised
from diayn_seq_code_revised.base.rollouter_base import RolloutWrapperBase, RollouterBase

import self_supervised.utils.typed_dicts as td

from rlkit.samplers.rollout_functions import rollout


class RlkitRolloutSamplerWrapper(RolloutWrapperBase):

    def __init__(self, rollout_fun=rollout):
        self.rollout_fun = rollout_fun

    def rollout(self,
                env: gym.Env,
                policy,
                seq_len=None):
        path = self.rollout_fun(
            env=env,
            agent=policy,
            max_path_length=seq_len
        )

        assert len(path['observations'].shape) == 2
        assert path['observations'].shape[-1] == env.observation_space.shape[0]

        return path

    def _reshape(self, path: dict):
        for k, v in path.items():
            if type(path[k]) is np.ndarray:
                path[k] = np.transpose(path[k])

        return path


class RollouterRevised(RollouterBase):

    def __init__(self,
                 *args,
                 policy: Union[SkillTanhGaussianPolicyRevised,
                               MakeDeterministicRevised],
                 rollout_wrapper: RolloutWrapperBase,
                 **kwargs
                 ):
        super(RollouterRevised, self).__init__(*args, **kwargs)

        self.policy = policy
        self.rollout_wrapper = rollout_wrapper


    def do_rollout(
            self,
            seq_len: int = None,
    ) -> td.TransitionMapping:
        """
        Args:
            seq_len         : if None seq_len is set to inf
        """
        path = self.rollout_wrapper.rollout(
            env=self.env,
            policy=self.policy,
            seq_len=seq_len,
        )

        return td.TransitionMapping(
            obs=path['observations'],
            action=path['actions'],
            reward=path['rewards'],
            next_obs=path['next_observations'],
            terminal=path['terminals'],
        )

    def reset(self):
        self.env.reset()
