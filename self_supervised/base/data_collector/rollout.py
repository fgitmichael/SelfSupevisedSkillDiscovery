import gym
from collections import deque,namedtuple
from prodict import Prodict
import numpy as np

from rlkit.samplers.data_collector.base import PathCollector
from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.policies.base import Policy
from rlkit.torch.sac.diayn.policies import SkillTanhGaussianPolicy


class PathMapping(Prodict):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    agent_infos: dict
    env_infos: dict

    def __init__(self,
                 observations: np.ndarray,
                 actions: np.ndarray,
                 rewards: np.ndarray,
                 next_observations: np.ndarray,
                 terminals: np.ndarray,
                 agent_infos: dict,
                 env_infos: dict):
        super(PathMapping, self).__init__(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            agent_infos=agent_infos,
            env_infos=env_infos
        )


class Rollouter(object):

    def __init__(self,
                 env: gym.Env,
                 policy: SkillTanhGaussianPolicy):
        self._env = env
        self._policy = policy

    def do_rollout(self,
                max_path_length: int=None,
                render: bool=None,
                render_kwargs: dict=None) -> PathMapping:

        path = rollout(
            env=self._env,
            agent=self._policy,
            max_path_length=max_path_length,
            render=render,
            render_kwargs=render_kwargs
        )

        path = self._reshape_path(path)

        return PathMapping(**path)

    @staticmethod
    def _reshape_path(path):
        assert len(path['rewards'].shape) == len(path['terminals'].shape) == 1

        path['rewards'] = np.expand_dims(path['rewards'], 1)
        path['terminals'] = np.expand_dims(path['terminals'], 1)

        return path
