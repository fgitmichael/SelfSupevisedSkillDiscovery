import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import numpy as np


def test_pybullet():
    env = gym.make('HopperPyBulletEnv-v0')
    env.render()
    env.reset()
    for _ in range(1000):
        env.step(env.action_space.sample())
        env.render()

test_pybullet()
