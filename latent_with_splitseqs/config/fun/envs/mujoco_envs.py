from gym.envs.mujoco.swimmer_v3 import SwimmerEnv as SwimmerVersionThreeEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv as HalfCheetahVersionThreeEnv
from gym.envs.mujoco.ant_v3 import AntEnv as AntVersionThreeEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv as HopperVersionThreeEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv as Walker2dVersionThreeEnv

from latent_with_splitseqs.config.fun.envs.locomotion_env_keys import locomotion_env_keys


gym_envs_version_three = {
    # 2d current position
    locomotion_env_keys['swimmer_key']: SwimmerVersionThreeEnv,
    # 1d current position
    locomotion_env_keys['halfcheetah_key']: HalfCheetahVersionThreeEnv,
    # 2d current position
    locomotion_env_keys['ant_key']: AntVersionThreeEnv,
    # 1d current position
    locomotion_env_keys['hopper_key']: HopperVersionThreeEnv,
    # 1d current position
    locomotion_env_keys['walker_key']: Walker2dVersionThreeEnv,
}
