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

        return PathMapping(**path)
