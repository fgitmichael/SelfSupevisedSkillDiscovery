import gym
import numpy as np

if __name__ == '__main__':
    #env = gym.make('MountainCarContinuous-v0')
    env = gym.make('HalfCheetah-v2')

    done = False
    env.reset()
    env.render()
    while not done:
        #obs, _, done, _ = env.step(env.action_space.sample())
        obs, _, done, _ = env.step(np.array([0.99]))
        env.render()

    env.close()
