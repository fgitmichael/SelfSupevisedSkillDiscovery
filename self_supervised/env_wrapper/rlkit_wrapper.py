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
                 gym_id: str,
                 action_repeat: int = 1,
                 obs_type: str ='state',
                 normalize_states: bool = True,
                 render_kwargs: bool = None):
        super().__init__(
            gym_id=gym_id,
            action_repeat=action_repeat,
            obs_type=obs_type,
            normalize_states=normalize_states,
            render_kwargs=render_kwargs
        )

    # Only change: put denormalization into env
    def step(self, action: np.ndarray) -> Tuple[ObsReturn, float, bool, dict]:
        next_obs, reward, done, info = super().step(action)

        obs_return = self.ObsReturn(
            normalized=next_obs,
            denormalized=self.denormalize(next_obs)
        )

        return obs_return, reward, done, info
