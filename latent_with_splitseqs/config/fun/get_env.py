from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper

from gym.envs.mujoco.swimmer_v3 import SwimmerEnv as SwimmerVersionThreeEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv as HalfCheetahVersionThreeEnv
from gym.envs.mujoco.ant_v3 import AntEnv as AntVersionThreeEnv

from two_d_navigation_demo.env.navigation_env import TwoDimNavigationEnv

gym_envs_version_three = dict(
    swimmer=SwimmerVersionThreeEnv,
    halfcheetah=HalfCheetahVersionThreeEnv,
    ant=AntVersionThreeEnv,
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

    # Keys
    gym_id_key = 'env_id'
    exclude_current_positions_key = 'exclude_current_positions_from_observation'
    assert gym_id_key in env_kwargs.keys()
    assert exclude_current_positions_key in env_kwargs.keys()

    if exclude_current_positions_key in env_kwargs.keys():
        exclude_current_positions_from_observation \
            = env_kwargs[exclude_current_positions_key]

    else:
        exclude_current_positions_from_observation = True

    # Return Environment
    gym_id = env_kwargs[gym_id_key]
    if gym_id in gym_envs_version_three.keys():
        env = gym_envs_version_three[gym_id](
            exclude_current_positions_from_observation= \
            exclude_current_positions_from_observation
        )

    elif gym_id in gym_envs_normal.keys():
        env = gym_envs_normal[gym_id]()

    else:
        env = NormalizedBoxEnvWrapper(env_kwargs[gym_id_key])

    return env