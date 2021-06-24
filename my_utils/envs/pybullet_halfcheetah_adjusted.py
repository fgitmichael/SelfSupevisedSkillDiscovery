import copy
from functools import wraps
from pybulletgym.envs.mujoco.envs.locomotion.half_cheetah_env \
    import HalfCheetahMuJoCoEnv
from pybulletgym.envs.mujoco.robots.locomotors.half_cheetah \
    import HalfCheetah
from pybulletgym.envs.mujoco.robots.robot_bases import MJCFBasedRobot
from pybulletgym.envs.mujoco.envs.locomotion.walker_base_env import WalkerBaseMuJoCoEnv


class HalfCheetahAdjusted(HalfCheetah):

    def __init__(
            self,
            xml_file_path: str,
    ):
        super().__init__()
        MJCFBasedRobot.__init__(
            self,
            xml_file_path,
            "torso",
            action_dim=6,
            obs_dim=17,
            add_ignored_joints=True
        )
        # self.pos_after = 0


def halfcheetah_env_creator(xml_file_path: str):
    robot = HalfCheetahAdjusted(
        xml_file_path=xml_file_path,
    )

    env_class = copy.deepcopy(HalfCheetahMuJoCoEnv)
    orig_init = env_class.__init__

    @wraps(orig_init)
    def new_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.robot = robot
        WalkerBaseMuJoCoEnv.__init__(self, self.robot)

    env_class.__init__ = new_init

    return env_class
