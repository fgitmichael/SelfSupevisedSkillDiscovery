import os

from pybulletgym.envs.mujoco.robots.locomotors.ant import Ant
from pybulletgym.envs.mujoco.robots.locomotors.hopper import Hopper
from pybulletgym.envs.mujoco.robots.locomotors.half_cheetah import HalfCheetah
from pybulletgym.envs.mujoco.robots.locomotors.walker2d import Walker2D
from pybulletgym.envs.mujoco.robots.robot_bases import MJCFBasedRobot

from latent_with_splitseqs.config.fun.envs.locomotion_env_keys import locomotion_env_keys


assets_base_path = os.path.join('..', 'my_utils', 'envs', 'assets')
env_xml_file_paths = {
    locomotion_env_keys['halfcheetah_key']: os.path.join(
        assets_base_path, 'half_cheetah_mujoco_version.xml'
    ),
    locomotion_env_keys['ant_key']: os.path.join(
        assets_base_path, 'ant_mujoco_version.xmal'
    ),
    locomotion_env_keys['walker_key']: os.path.join(
        assets_base_path, 'walker_mujoco_version.xmal'
    ),
    locomotion_env_keys['hopper_key']: os.path.join(
        assets_base_path, 'hopper_mujoco_version.xmal'
    ),
}


class HalfCheetahAdjusted(HalfCheetah):

    def __init__(
            self,
            xml_file_path: str,
    ):
        super().__init__()
        assert os.path.isfile(xml_file_path)
        MJCFBasedRobot.__init__(
            self,
            xml_file_path,
            "torso",
            action_dim=6,
            obs_dim=17,
            add_ignored_joints=True
        )
        # self.pos_after = 0


class HopperAdjusted(Hopper):

    def __init__(self, xml_file_path: str):
        super().__init__()
        MJCFBasedRobot.__init__(
            self,
            xml_file_path,
            "torso",
            action_dim=3,
            obs_dim=11,
            add_ignored_joints=True
        )


class AntAdjusted(Ant):

    def __init__(self, xml_file_path: str):
        super().__init__()
        MJCFBasedRobot.__init__(
            self,
            xml_file_path,
            "torso",
            action_dim=8,
            obs_dim=111
        )


class Walker2DAdjusted(Walker2D):

    def __init__(self, xml_file_path: str):
        super().__init__()
        MJCFBasedRobot.__init__(
            self,
            xml_file_path,
            "torso",
            action_dim=6,
            obs_dim=17,
            add_ignored_joints=True
        )


robots_adjusted = {
    locomotion_env_keys['halfcheetah_key']: HalfCheetahAdjusted,
    locomotion_env_keys['hopper_key']: HopperAdjusted,
    locomotion_env_keys['walker_key']: Walker2DAdjusted,
    locomotion_env_keys['ant_key']: AntAdjusted,
}

