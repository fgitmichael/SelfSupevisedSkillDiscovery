import torch

from mode_disent.test.action_sampler import ActionSamplerWithActionModel
from mode_disent.test.interactive_disent_testing import InteractiveDisentTester
from code_slac.env.ordinary_env import OrdinaryEnvForPytorch

# Note: This script has to be run in the model-log folder

def run():
    # Load viz base figure
    axes_path_mode_mapping = './mode_mapping.axes'
    fig_path_mode_mapping = './mode_mapping.fig'
    fig = torch.load(fig_path_mode_mapping)

    # Load models
    dyn_model = torch.load('./dyn_model.pth')
    mode_model = torch.load('./mode_model.pth')

    # Environment
    env = OrdinaryEnvForPytorch('MountainCarContinuous-v0')

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


