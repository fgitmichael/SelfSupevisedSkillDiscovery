import os
import torch
import gym

from mode_disent_no_ssm.utils.parse_args import load_hparams

from my_utils.dicts.get_config_item import get_config_item

import latent_with_splitseqs.config.fun.get_env as get_env


def _load_config(dir: str) -> dict:
    # Get file list of summary folder
    summary_dir_list = os.listdir(dir)

    # Get yaml file(s)
    extension = ".yaml"
    yaml_files = [
        filestr
        for filestr in summary_dir_list
        if os.path.splitext(filestr)[-1] == extension
    ]
    assert len(yaml_files) == 1, "More than one yaml config file found"

    # Load config
    config = load_hparams(os.path.join(dir, yaml_files[0]))

    return config


def load_env(dir='../summary') -> gym.Env:
    # Load config
    config = _load_config(dir=dir)

    # Load environment
    env_is_pybullet = get_config_item(
        config=config,
        key=['env_kwargs', get_env.pybullet_key, get_env.is_pybullet_key],
        default=False,
    )
    if env_is_pybullet:
        print("Pybullet")
        # Serialized py bullet envs can't be loaded and need be created again
        env = get_env.get_env(**config['env_kwargs'])
    else:
        print("Mujoco")
        extension = ".pkl"
        env_name = "env" + extension
        env = torch.load(env_name)

    return env
