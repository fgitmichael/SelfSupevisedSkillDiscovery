import gym
from collections import deque,namedtuple
from prodict import Prodict
import numpy as np

from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.sac.diayn.policies import SkillTanhGaussianPolicy

from self_supervised.utils.typed_dicts import TransitionMapping


class Rollouter(object):

    def __init__(self,
                 env: gym.Env,
                 policy: SkillTanhGaussianPolicy):
        self._env = env
        self._policy = policy

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
