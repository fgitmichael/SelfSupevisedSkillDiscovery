import gym

from my_utils.dicts.get_config_item import get_config_item

from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper

from my_utils.dicts.remove_nones import remove_nones

from two_d_navigation_demo.env.navigation_env import TwoDimNavigationEnv

from latent_with_splitseqs.config.fun.envs.locomotion_env_keys import locomotion_env_keys
from latent_with_splitseqs.config.fun.envs.pybullet_envs \
    import pybullet_envs_version_three, wrap_env_class

gym_envs_normal = dict(
    two_d_nav=TwoDimNavigationEnv,
)

gym_id_key = 'env_id'
init_kwargs_key = 'init_kwargs'
pybullet_key = 'pybullet'
is_pybullet_key = 'is_pybullet'
pos_dim_key = 'pos_dim'


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

        else:
            # PyBullet
            env_class = wrap_env_class(
                env_class_in=pybullet_envs_version_three[gym_id],
                pos_dim=get_config_item(
                    env_kwargs[pybullet_key],
                    key=pos_dim_key,
                    default=None
                )
            )
            env = env_class(**init_kwargs)

    elif gym_id in gym_envs_normal.keys():
        env = gym_envs_normal[gym_id](
            action_max=(0.01, 0.01)
        )

    else:
        env = NormalizedBoxEnvWrapper(env_kwargs[gym_id_key])

    return env
