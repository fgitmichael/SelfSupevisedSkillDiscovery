import numpy as np
from typing import Tuple, Dict

from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper


class PixelNormalizedBoxEnvWrapper(NormalizedBoxEnvWrapper):
    def __init__(self,
                 *args,
                 render_kwargs=None,
                 **kwargs):
        super(PixelNormalizedBoxEnvWrapper, self).__init__(
            *args,
            **kwargs,
        )
        self.render_kwargs = render_kwargs

    def step(self,
             action: np.ndarray,
             denormalize: bool = False,
             ) -> Tuple:
        step_tuple = super().step(
            action=action,
        )
        if self.render_kwargs is None:
            pixel_obs = self.env.render(mode='rgb_array')
        else:
            pixel_obs = self.env.render(
                mode='rgb_array',
                **self.render_kwargs
            )

        next_obs = dict(
            state_obs=step_tuple[0],
            pixel_obs=pixel_obs,
        )

        return (next_obs, *step_tuple[1:])

    def reset(self):
        obs = super().reset()
        if self.render_kwargs is None:
            pixel_obs = self.env.render(mode='rgb_array')
        else:
            pixel_obs = self.env.render(
                mode='rgb_array',
                **self.render_kwargs,
            )

        return (obs, pixel_obs)
