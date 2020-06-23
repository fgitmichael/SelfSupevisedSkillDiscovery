import gym

if __name__ == '__main__':
    env = gym.make('HalfCheetah-v2')

    done = False
    env.reset()
    env.render()
    while not done:
        obs, _, done, _ = env.step(env.action_space.sample())
        env.render()
