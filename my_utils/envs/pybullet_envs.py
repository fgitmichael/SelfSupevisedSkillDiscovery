from pybulletgym.envs.mujoco.envs.locomotion.ant_env import AntMuJoCoEnv
from pybulletgym.envs.mujoco.envs.locomotion.half_cheetah_env import HalfCheetahMuJoCoEnv
from pybulletgym.envs.mujoco.envs.locomotion.hopper_env import HopperMuJoCoEnv
from pybulletgym.envs.mujoco.envs.locomotion.walker2d_env import Walker2DMuJoCoEnv

from latent_with_splitseqs.config.fun.envs.locomotion_env_keys import locomotion_env_keys


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


pybullet_envs_version_three = {
    locomotion_env_keys['swimmer_key']: SwimmerBulletVersionThreeEnv,
    locomotion_env_keys['halfcheetah_key']: HalfCheetahBulletVersionThreeEnv,
    locomotion_env_keys['ant_key']: AntBulletVersionThreeEnv,
    locomotion_env_keys['walker_key']: Walker2dBulletVersionThreeEnv,
    locomotion_env_keys['hopper_key']: HopperBulletVersionThreeEnv,
}