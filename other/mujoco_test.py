#PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/michael/.mujoco/mujoco200/bin;LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
import gym
import os
env = gym.make('HalfCheetah-v2')
env.render()
px = env.render(mode='rgb_array')
print(px)
print(px.shape)
