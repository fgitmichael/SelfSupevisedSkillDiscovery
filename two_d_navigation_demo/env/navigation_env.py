import gym
from gym.utils import seeding
import numpy as np

class TwoDimNavigationEnv(gym.Env):

    def __init__(self,
                 size: tuple = (1,1),
                 action_max: tuple = (0.1, 0.1)
                 ):
        assert len(action_max) == 2
        assert len(size) == 2

        self.min_observation = np.zeros((2,), dtype=np.float)
        self.max_observation = np.array(size, dtype=np.float)

        self.min_action = np.zeros((2,), dtype=np.float)
        self.max_action = np.array(action_max, dtype=np.float)

        self.action_space = gym.spaces.Box(
            low=self.min_action,
            high=self.max_action,
            dtype=np.float,
        )

        self.observation_space = gym.spaces.Box(
            low=self.min_observation,
            high=self.max_observation,
            dtype=np.float,
        )

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.mean(
            self.observation_space.high - self.observation_space.low,
            axis=-1)
        return self.state

    def step(self, action: np.ndarray):
        self.state += action
        self.state = self.map_back(self.state)
        reward = 0
        done = False
        info = {}
        return self.state, reward, done, {}

    def map_back(self, state):
        assert state.shape \
               == self.min_observation.shape \
               == self.max_observation.shape
        return np.clip(
            state,
            a_min=self.min_observation,
            a_max=self.max_observation,
        )
