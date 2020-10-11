from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper

from gym.envs.mujoco.swimmer_v3 import SwimmerEnv as SwimmerVersionThreeEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv as HalfCheetahVersionThreeEnv
from gym.envs.mujoco.ant_v3 import AntEnv as AntVersionThreeEnv


def get_env(**env_kwargs):
    """
    Args:
        env_kwargs
            gym_id                                              : str name of environment
            exclude_current_positions_from_observation          : bool
    """
    gym_id_key = 'env_id'
    exclude_current_positions_key = 'exclude_current_positions_from_observation'
    assert gym_id_key in env_kwargs.keys()

    gym_envs_version_three = dict(
        simmer=SwimmerVersionThreeEnv,
        halfcheetah=HalfCheetahVersionThreeEnv,
        ant=AntVersionThreeEnv,
    )

    if exclude_current_positions_key in env_kwargs.keys():
        exclude_current_positions_from_observation \
            = env_kwargs[exclude_current_positions_key]
    else:
        exclude_current_positions_from_observation = None

    if env_kwargs[gym_id_key] in gym_envs_version_three.keys():
        return gym_envs_version_three[gym_id_key](
            exclude_current_positions_from_observation= \
            exclude_current_positions_from_observation
        )

    else:
        return NormalizedBoxEnvWrapper(env_kwargs[gym_id_key])
