import gym
from gym.utils import seeding
import numpy as np
import warnings


class OneDimNavigationEnv(gym.Env):

    def __init__(self,
                 size: float=10.,
                 action_max: float=1.,
                 ):
        self.min_obs = np.zeros((1,), dtype=np.float)
        self.max_obs = np.array([size], dtype=np.float)

        self.max_action = np.array([action_max], dtype=np.float)
        self.min_action = -self.max_action

        self.action_space = gym.spaces.Box(
            low=self.min_action,
            high=self.max_action,
            dtype=np.float,
        )

        self.observation_space = gym.spaces.Box(
            low=self.min_obs,
            high=self.max_obs,
            dtype=np.float,
        )

        self.seed()

        self.state = None
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.mean(
            np.stack([self.observation_space.low,
                      self.observation_space.high],
                     axis=-1),
            axis=-1
        )
        assert self.state in self.observation_space
        return self.state

    def step(self, action: np.ndarray):
        self.check_action(action)
        self.state = self.state + action
        self.state = self.map_back(self.state)
        reward = 0.
        done = False
        info = {}
        return self.state, reward, done, {}

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass

    def map_back(self, state):
        if state not in self.observation_space:
            mapped_back = np.clip(
                state,
                a_min=self.observation_space.low,
                a_max=self.observation_space.high,
            )
        else:
            mapped_back = state
        assert mapped_back in self.observation_space

        return mapped_back

    def check_action(self, action: np.ndarray):
        if action not in self.action_space \
                and np.abs(action) - (self.action_space.high - self.action_space.low) < 1E7:
            warnings.warn('Action is numerically out of actions space')

        elif np.abs(action) - (self.action_space.high - self.action_space.low) > 1E7:
            raise ValueError('Action is out of action space')
