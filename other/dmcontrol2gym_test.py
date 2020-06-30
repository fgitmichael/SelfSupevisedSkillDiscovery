import gym

# make the dm_control environment
env = gym.make('dm2gym:CheetahRun-v0')

# use same syntax as in gym
env.reset()
for t in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    env.render()

