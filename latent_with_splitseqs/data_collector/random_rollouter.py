import numpy as np
import gym

from diayn_seq_code_revised.base.rollouter_base import RollouterBase

import self_supervised.utils.typed_dicts as td

from rlkit.samplers.rollout_functions import rollout


class RollouterRandom(RollouterBase):

    def __init__(self,
                 env: gym.Env,
                 rollout_fun=rollout,
                 ):
        self.env = env
        self.random_action_generator = RandomActionGenerator(
            action_space=env.action_space
        )
        self.rollout_fun = rollout_fun

    def do_rollout(
            self,
            seq_len: int) -> td.TransitionMapping:

        path = self.rollout_fun(
            env=self.env,
            agent=self.random_action_generator,
            max_path_length=seq_len,
        )
        assert len(path['observations'].shape) == 2
        assert path['observations'].shape[-1] == self.env.observation_space.shape[0]

        return td.TransitionMapping(
            obs=path['observations'],
            action=path['actions'],
            reward=path['rewards'],
            next_obs=path['next_observations'],
            terminal=path['terminals'],
        )

    def reset(self):
        self.env.reset()


class RandomActionGenerator:

    def __init__(self, action_space: gym.Space):
        self.action_shape = action_space

    def get_action(self, obs):
        action = self.action_shape.sample()

        return action, {}

    def reset(self):
        pass
