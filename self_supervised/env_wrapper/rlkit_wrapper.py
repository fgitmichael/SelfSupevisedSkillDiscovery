from prodict import Prodict
from typing import Tuple, Union
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
                 *args,
                 **env_kwargs,
                 ):
        super().__init__(
            *args,
            **env_kwargs
        )

    # Only change: put denormalization into env
    def step(self,
             action: np.ndarray,
             denormalize: bool = False) \
            -> Tuple[np.ndarray, float, bool, dict]:
        next_obs, reward, done, info = super().step(action)

        if denormalize:
            obs_return = self.denormalize(next_obs)
        else:
            obs_return = next_obs

        return obs_return, reward, done, info

    @property
    def obs_stats(self):
        return dict(
            mean=self.obs_mean,
            std=self.obs_std
        )

    @obs_stats.setter
    def obs_stats(self, obs_stats):
        self.obs_mean = obs_stats['mean']
        self.obs_std = obs_stats['std']
