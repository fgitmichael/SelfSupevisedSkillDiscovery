import numpy as np
import gym
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv


class HalfCheetahPixelWrapper:

    def __init__(self,
                 env: HalfCheetahEnv,
                 render_kwargs=None,
                 ):
        self.env = env
        self.render_kwargs=render_kwargs
        self.env.render()

    def get_pixel_obs(self):
        if self.render_kwargs is None:
            pixel_obs = self.env.render(mode='rgb_array')
        else:
            pixel_obs = self.env.render(
                mode='rgb_array',
                **self.render_kwargs
            )
        return pixel_obs

    def step(self,
             action: np.ndarray,
             ):
        step_tuple = self.env.step(
            action=action
        )
        pixel_obs = self.get_pixel_obs()

        next_obs = dict(
            state_obs=step_tuple[0],
            pixel_obs=pixel_obs
        )

        return (next_obs, *step_tuple[1:])

    def reset(self):
        obs = self.reset()
        pixel_obs = self.get_pixel_obs()

        return dict(
            state_obs=obs,
            pixel_obs=pixel_obs,
        )
