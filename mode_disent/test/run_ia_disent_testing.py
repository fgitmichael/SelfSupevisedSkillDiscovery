import torch
import json
from easydict import EasyDict as edict

from mode_disent.test.action_sampler import ActionSamplerWithActionModel
from mode_disent.test.interactive_disent_testing import InteractiveDisentTester
from code_slac.env.ordinary_env import OrdinaryEnvForPytorch
from code_slac.env.dm_control import DmControlEnvForPytorch

# Note: This script has to be run in the model-log folder


def load_hparams(file_path):
    f = open(file_path)
    hparams_dict = json.load(f)
    return edict(hparams_dict)


def run():
    # Load run hyperparameter
    hparams = load_hparams('run_hyperparameter.json')

    # Load viz base figure
    axes_path_mode_mapping = './mode_mapping.axes'
    fig_path_mode_mapping = './mode_mapping.fig'
    fig = torch.load(fig_path_mode_mapping)

    # Load models
    dyn_model = torch.load('./dyn_model.pth')
    mode_model = torch.load('./mode_model.pth')

    # Environment
    env = OrdinaryEnvForPytorch('MountainCarContinuous-v0')
    if hparams.env_info.env_type == 'normal':
        env = OrdinaryEnvForPytorch(hparams.env_info.env_id)
    elif hparams.env_info.env_type == 'dm_control':
        env = DmControlEnvForPytorch(
            domain_name=hparams.env_info.domain_name,
            task_name=hparams.env_info.task_name,
            action_repeat=hparams.env_info.action_repeat,
            obs_type=hparams.env_info.obs_type
        )

    tester = InteractiveDisentTester(
        dyn_model=dyn_model,
        mode_model=mode_model,
        device='cuda',
        env=env,
        mode_map_fig=fig,
        num_episodes=1000,
        seed=0
    )

    tester.run()


if __name__ == '__main__':
    run()


