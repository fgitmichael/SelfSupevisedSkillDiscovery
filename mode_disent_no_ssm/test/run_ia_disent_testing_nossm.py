import torch
from easydict import EasyDict as edict

from mode_disent_no_ssm.utils.parse_args import load_hparams
from mode_disent_no_ssm.test.interactive_disent_tester \
    import InteractiveDisentTester as IaDisentTesterNoSSM
from mode_disent.env_wrappers.rlkit_wrapper import NormalizedBoxEnvForPytorch
# Note: This script has to be run in the model-log folder
#       Set path variable for Mujoco using ...
#       LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/michael/.mujoco/mujoco200/bin
#       Call using


def run(device='cpu'):
    hparams = load_hparams('run_hyperparameters.json')
    hparams = edict(hparams)

    fig_path_mode_mapping = './mode_mapping.fig'
    fig = torch.load(fig_path_mode_mapping)

    mode_model = torch.load('./mode_model.pkl', map_location=device)
    obs_encoder = torch.load('./obs_encoder.pkl', map_location=device)
    mode_model.device = device
    obs_encoder.device = device

    env = NormalizedBoxEnvForPytorch(
        gym_id=hparams.env_info.env_id,
        action_repeat=hparams.env_info.action_repeat,
        obs_type='state',
        normalize_states=True
    )

    tester = IaDisentTesterNoSSM(
        mode_model=mode_model,
        obs_encoder=obs_encoder,
        device=device,
        env=env,
        mode_map_fig=fig,
        num_episodes=1000,
        len_sequence=hparams.num_sequences * 2,
        seed=0
    )

    tester.run()

if __name__ == '__main__':
    run(device='cpu')

