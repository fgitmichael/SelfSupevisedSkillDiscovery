import numpy as np
from gym.spaces import Box
from typing import Type

from my_utils.dicts.get_config_item import get_config_item

from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv

env_kwargs_keys = dict(
    exclude_current_position_key='exclude_current_positions_from_observation',
    reset_noise_scale='reset_noise_scale',
)


def get_current_position(self) -> np.ndarray:
    """
    Pybullet does not seem to carry the current position
    consistently for all environments.
    In the Halfcheetah environment, for example, the current position is kept
    as "position_after"-attribute, while in the hopper environment the
    current position can only be retrieved via robotxyz. However robotxzy does not work
    for halfcheetah, it always shows zero no matter what happens with the robot.
    These difference are handled in this function.
    """
    pos_after_attr_key = 'pos_after'
    if hasattr(self.robot, pos_after_attr_key):
        pos_after = getattr(self.robot, pos_after_attr_key)
        assert isinstance(pos_after, float)
        ret_val = [pos_after,]

    else:
        ret_val = self.robot.body_xyz

    return np.array(ret_val)

"""
Class Wrapper
"""
def _add_position(
        obs: np.ndarray,
        current_position: np.ndarray,
        pos_dim: int
) -> np.ndarray:
    assert len(obs.shape) == 1
    obs_dim = obs.shape[0]

    # Add position
    current_observation_to_add = current_position[:pos_dim]
    obs_with_pos = np.insert(obs, 0, current_observation_to_add)
    assert obs_with_pos.shape[0] == obs_dim + pos_dim

    return obs_with_pos


def wrap_env_class(
        env_class_in: Type[BaseBulletEnv],
        pos_dim: int,
) -> Type[BaseBulletEnv]:
    """
    Wrapper function to add current position to the observation
        env_class           : environement class
        pos_dim             : dimension of position
    """
    class EnvClassOut(env_class_in):

        def __init__(self, *args, **kwargs):
            # Get position exclude bool
            self.exclude_current_position = get_config_item(
                kwargs,
                key=env_kwargs_keys['exclude_current_position_key'],
                default=True,
            )

            # Check for no reset_noise scaling
            reset_noise_scale = get_config_item(
                kwargs,
                key=env_kwargs_keys['reset_noise_scale'],
                default=None,
            )
            assert reset_noise_scale is None

            # Delete keys
            for key_str in env_kwargs_keys.values():
                if key_str in kwargs.keys():
                    del kwargs[key_str]

            # Call original init method
            super().__init__(*args, **kwargs)

            # Adjust observation space if position is not excluded
            assert "observation_space" in vars(self).keys()
            orig_obs_space = self.observation_space
            if not self.exclude_current_position:
                assert isinstance(orig_obs_space, Box)
                low = orig_obs_space.low
                high = orig_obs_space.high
                low_new = np.concatenate([-np.inf * np.ones([pos_dim]), low])
                high_new = np.concatenate([np.inf * np.ones([pos_dim]), high])
                self.observation_space = Box(low_new, high_new)

        def reset(self):
            obs = super().reset()
            if not self.exclude_current_position:
                current_position = get_current_position(self)
                obs = _add_position(
                    obs=obs,
                    current_position=current_position,
                    pos_dim=pos_dim,
                )
            return obs

        def step(self, *args, **kwargs):
            # Get observation
            step_return = super().step(*args, **kwargs)
            step_return = list(step_return)
            obs = step_return[0]

            if not self.exclude_current_position:
                current_position = get_current_position(self)
                # Add position
                step_return[0] = _add_position(
                    obs=obs,
                    current_position=current_position,
                    pos_dim=pos_dim,
                )

            return step_return

    return EnvClassOut
