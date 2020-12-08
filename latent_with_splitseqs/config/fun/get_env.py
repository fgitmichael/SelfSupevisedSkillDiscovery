from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper

from gym.envs.mujoco.swimmer_v3 import SwimmerEnv as SwimmerVersionThreeEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv as HalfCheetahVersionThreeEnv
from gym.envs.mujoco.ant_v3 import AntEnv as AntVersionThreeEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv as HopperVersionThreeEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv as Walker2dVersionThreeEnv

from my_utils.dicts.remove_nones import remove_nones

from two_d_navigation_demo.env.navigation_env import TwoDimNavigationEnv


gym_envs_version_three = dict(
    # 2d current position
    swimmer=SwimmerVersionThreeEnv,
    # 1d current position
    halfcheetah=HalfCheetahVersionThreeEnv,
    # 2d current position
    ant=AntVersionThreeEnv,
    # 1d current position
    hopper=HopperVersionThreeEnv,
    # 1d current position
    walker=Walker2dVersionThreeEnv,
)

gym_envs_normal = dict(
    two_d_nav=TwoDimNavigationEnv,
)


def get_env(**env_kwargs):
    """
    Args:
        env_kwargs
            env_id                                              : str name of environment
            exclude_current_positions_from_observation          : bool
    """
    global gym_envs_version_three
    global gym_envs_normal

    gym_id_key = 'env_id'
    init_kwargs_key = 'init_kwargs'
    init_kwargs = env_kwargs[init_kwargs_key]
    init_kwargs = remove_nones(init_kwargs)

    # Return Environment
    gym_id = env_kwargs[gym_id_key]
    if gym_id in gym_envs_version_three.keys():
        env = gym_envs_version_three[gym_id](**init_kwargs)

    elif gym_id in gym_envs_normal.keys():
        env = gym_envs_normal[gym_id]()

    else:
        env = NormalizedBoxEnvWrapper(env_kwargs[gym_id_key])

    return env
