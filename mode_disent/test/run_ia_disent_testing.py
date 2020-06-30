import torch
import json
from easydict import EasyDict as edict

from mode_disent.test.action_sampler import ActionSamplerWithActionModel
from mode_disent.test.interactive_disent_testing import InteractiveDisentTester
from mode_disent.env_wrappers.dmcontrol import MyDmControlEnvForPytorch
from mode_disent.env_wrappers.rlkit_wrapper import NormalizedBoxEnvForPytorch
from code_slac.env.ordinary_env import OrdinaryEnvForPytorch
from code_slac.env.dm_control import DmControlEnvForPytorch

# Note: This script has to be run in the model-log folder
#       Set path variable for Mujoco using ...
#       LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/michael/.mujoco/mujoco200/bin
#       Call using


def load_hparams(file_path):
    f = open(file_path)
    hparams_dict = json.load(f)
    return edict(hparams_dict)


def run(device='cpu'):
    # Load run hyperparameter
    hparams = load_hparams('run_hyperparameter.json')

    # Load viz base figure
    fig_path_mode_mapping = './mode_mapping.fig'
    fig = torch.load(fig_path_mode_mapping)

    # Load models
    dyn_model = torch.load('./dyn_model.pth', map_location=device)
    mode_model = torch.load('./mode_model.pth', map_location=device)

    # Environment
    obs_type = "state" if hparams.state_rep is True else "pixels"
    if hparams.env_info.env_type == 'normal':
        env = OrdinaryEnvForPytorch(hparams.env_info.env_id)

    elif hparams.env_info.env_type == 'dm_control':
        env = MyDmControlEnvForPytorch(
            domain_name=hparams.env_info.domain_name,
            task_name=hparams.env_info.task_name,
            action_repeat=hparams.env_info.action_repeat,
            obs_type=hparams.env_info.obs_type
        )

    elif hparams.env_info.env_type == 'normalized':
        env = NormalizedBoxEnvForPytorch(
            gym_id=hparams.env_info.env_id,
            action_repeat=hparams.env_info.action_repeat,
            obs_type=obs_type,
            normalize_states=hparams.env_info.normalize_states,
        )

    else:
        raise ValueError('Env_type is not used in if else statesments')

    tester = InteractiveDisentTester(
        dyn_model=dyn_model,
        mode_model=mode_model,
        device=device,
        env=env,
        mode_map_fig=fig,
        num_episodes=1000,
        len_sequence=hparams.num_sequences * 2,
        seed=0
    )

    tester.run()


if __name__ == '__main__':
    run()


