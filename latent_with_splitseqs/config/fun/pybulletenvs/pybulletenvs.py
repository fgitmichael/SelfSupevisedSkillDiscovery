import numpy as np
from functools import wraps
from typing import Type

from my_utils.dicts.get_config_item import get_config_item

from pybulletgym.envs.mujoco.envs.locomotion.hopper_env import HopperMuJoCoEnv
from pybulletgym.envs.mujoco.envs.locomotion.ant_env import AntMuJoCoEnv
from pybulletgym.envs.mujoco.envs.locomotion.half_cheetah_env import HalfCheetahMuJoCoEnv
from pybulletgym.envs.mujoco.envs.locomotion.walker2d_env import Walker2DMuJoCoEnv
from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv

env_kwargs_keys = dict(
    exclude_current_position_key='exclude_current_positions_from_observation',
    reset_noise_scale='reset_noise_sacle'
)


"""
Mujoco-like envs
"""
class SwimmerBulletVersionThreeEnv(object):

    def __init__(self):
        raise NotImplementedError


class HalfCheetahBulletVersionThreeEnv(HalfCheetahMuJoCoEnv):
    pass


class AntBulletVersionThreeEnv(AntMuJoCoEnv):
    pass


class HopperBulletVersionThreeEnv(HopperMuJoCoEnv):
    pass


class Walker2dBulletVersionThreeEnv(Walker2DMuJoCoEnv):
    pass


pybullet_envs_version_three = dict(
    swimmer=SwimmerBulletVersionThreeEnv,
    halfcheetah=HalfCheetahBulletVersionThreeEnv,
    ant=AntBulletVersionThreeEnv,
    walker=Walker2dBulletVersionThreeEnv,
)


"""
Class Wrapper
"""
def get_env_wrapper(
        env_class: Type[BaseBulletEnv],
        pos_dim: int,
) -> Type[BaseBulletEnv]:
    """
    Wrapper function to add current position to the observation
        env_class           : environement class
        pos_dim             : dimension of position
    """
    class BaseBulletEnvCopy(env_class):
        pass
    orig_init = BaseBulletEnvCopy.__init__
    orig_step = BaseBulletEnvCopy.step

    @wraps(orig_init)
    def new_init(self, *args, **kwargs):
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
            del kwargs[key_str]

        # Call original init method
        orig_init(*args, **kwargs)

    @wraps(orig_step)
    def new_step(self, *args, **kwargs):
        # Get observation
        step_return = orig_step(*args, **kwargs)

        if not self.exclude_current_position:
            obs = step_return[0]
            assert len(obs.shape) == 1
            obs_dim = obs.shape[0]

            # Add position
            current_position = self.robot.body_xyz
            current_observation_to_add = current_position[:pos_dim]
            obs_with_pos = np.insert(obs, 0, current_observation_to_add)
            assert obs_with_pos.shape[0] == obs_dim + pos_dim

            # Replace observation
            step_return[0] = obs_with_pos

        return step_return

    BaseBulletEnvCopy.__init__ = new_init
    BaseBulletEnvCopy.step = new_step
    return BaseBulletEnvCopy
