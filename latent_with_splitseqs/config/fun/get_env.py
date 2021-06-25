import gym
import os

from my_utils.dicts.get_config_item import get_config_item

from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.utils.writer import model_folder_name, summary_folder_name

from my_utils.dicts.remove_nones import remove_nones

from two_d_navigation_demo.env.navigation_env import TwoDimNavigationEnv

from latent_with_splitseqs.config.fun.envs.locomotion_env_keys import locomotion_env_keys
from latent_with_splitseqs.config.fun.envs.pybullet_envs import \
    env_xml_file_paths, \
    pybullet_envs_version_three, \
    pybullet_envs_version_three_xml_change, \
    wrap_env_class
from latent_with_splitseqs.config.fun.envs.action_repeat_wrapper \
    import wrap_env_action_repeat

gym_envs_normal = dict(
    two_d_nav=TwoDimNavigationEnv,
)

gym_id_key = 'env_id'
init_kwargs_key = 'init_kwargs'
pybullet_key = 'pybullet'
is_pybullet_key = 'is_pybullet'
pos_dim_key = 'pos_dim'
change_xml_key = 'mujoco_physics'


def get_env(**env_kwargs) -> gym.Env:
    """
    Args:
        env_kwargs
            env_id                                              : str name of environment
            exclude_current_positions_from_observation          : bool
    """
    global gym_envs_version_three
    global gym_envs_normal

    init_kwargs = env_kwargs[init_kwargs_key]
    init_kwargs = remove_nones(init_kwargs)

    action_repeat = get_config_item(env_kwargs, key='action_repeat', default=1)

    # Return Environment
    gym_id = env_kwargs[gym_id_key]
    if gym_id in locomotion_env_keys.values():
        if not get_config_item(
                env_kwargs[pybullet_key],
                key=is_pybullet_key,
                default=False
        ):
            # MuJoCo
            # Conditional import to avoid unnecessary license checks
            from latent_with_splitseqs.config.fun.envs.mujoco_envs  \
                import gym_envs_version_three
            env = gym_envs_version_three[gym_id](**init_kwargs)

            #TODO: Check if all dimension are used, as if position is not
            # excluded the observation space of the mujoco envs stays the same
            raise ValueError('remaining TODO is left!')

        else:
            change_xml = get_config_item(
                env_kwargs['pybullet'],
                key=change_xml_key,
                default=False
            )
            if change_xml:
                # Get xml path, use either original or saved version in the summary folder
                cwd = os.getcwd()
                if os.path.basename(cwd) == model_folder_name:
                    xml_path = os.path.join(
                        "..",
                        summary_folder_name,
                        os.path.basename(env_xml_file_paths[gym_id]),
                    )

                else:
                    xml_path = env_xml_file_paths[gym_id]

                env_class_in = pybullet_envs_version_three_xml_change[gym_id](
                    os.path.abspath(xml_path)
                )

            else:
                env_class_in = pybullet_envs_version_three[gym_id]

            # PyBullet
            env_class = wrap_env_class(
                env_class_in=env_class_in,
                pos_dim=get_config_item(
                    env_kwargs[pybullet_key],
                    key=pos_dim_key,
                    default=None
                )
            )
            if action_repeat > 1:
                env_class = wrap_env_action_repeat(
                    env_class_in=env_class,
                    num_action_repeat=action_repeat,
                )
            env = env_class(**init_kwargs)

    elif gym_id in gym_envs_normal.keys():
        assert action_repeat == 1
        env = gym_envs_normal[gym_id](
            action_max=(0.01, 0.01)
        )

    else:
        env = NormalizedBoxEnvWrapper(
            env_kwargs[gym_id_key],
            action_repeat=action_repeat,
        )

    return env
