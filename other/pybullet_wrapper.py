import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import numpy as np

from code_slac.env.ordinary_env import OrdinaryEnvForPytorch


class PyBulletEnvForPytorch(OrdinaryEnvForPytorch):
    keys = ['state', 'pixels']




def test_pybullet():
    env = gym.make('HalfCheetahPyBulletEnv-v0')
    env.render()
    env.reset()
    for _ in range(1000):
        env.step(env.action_space.sample())
