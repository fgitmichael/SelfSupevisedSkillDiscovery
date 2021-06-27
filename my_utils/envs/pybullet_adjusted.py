from pybulletgym.envs.mujoco.envs.locomotion.walker_base_env \
    import WalkerBaseMuJoCoEnv

from my_utils.envs.robots_adjusted import robots_adjusted

from my_utils.envs.pybullet_envs import pybullet_envs_version_three


def env_creator_adjusted(
        gym_id,
        xml_file_path: str
):
    robot_class = robots_adjusted[gym_id]
    env_class = pybullet_envs_version_three[gym_id]
    robot = robot_class(
        xml_file_path=xml_file_path,
    )
    class EnvClassOut(env_class):
        def __init__(self):
            super().__init__()
            self.robot = robot
            WalkerBaseMuJoCoEnv.__init__(self, self.robot)

    return EnvClassOut
