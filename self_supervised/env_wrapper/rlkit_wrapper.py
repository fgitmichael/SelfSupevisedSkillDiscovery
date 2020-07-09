from prodict import Prodict
from typing import Tuple
import numpy as np

from mode_disent.env_wrappers.rlkit_wrapper import NormalizedBoxEnvForPytorch


class NormalizedBoxEnvWrapper(NormalizedBoxEnvForPytorch):

    class ObsReturn(Prodict):
        normalized: np.ndarray
        denormalized: np.ndarray

        def __init__(self,
                     normalized: np.ndarray,
                     denormalized: np.ndarray):
            super().__init__(
                normalized=normalized,
                denormalized=denormalized
            )

    # Only change: default for normalize states
    def __init__(self,
                 **env_kwargs,
                 ):
        super().__init__(
            **env_kwargs
        )

    # Only change: put denormalization into env
    def step(self, action: np.ndarray) -> Tuple[ObsReturn, float, bool, dict]:
        next_obs, reward, done, info = super().step(action)

        obs_return = self.ObsReturn(
            normalized=next_obs,
            denormalized=self.denormalize(next_obs)
        )

        return obs_return, reward, done, info
