import torch

from mode_disent.test.action_sampler import ActionSamplerWithActionModel
from mode_disent.test.interactive_disent_testing import InteractiveDisentTester
from code_slac.env.ordinary_env import OrdinaryEnvForPytorch

def run():
    # Load viz base figure
    axes_path_mode_mapping = './models/05_mode_mapping_fig/' \
                             'mode_disent1-20200608-2353mode_mapping.axes'
    fig_path_mode_mapping ='./models/05_mode_mapping_fig/' \
                           'mode_disent1-20200608-2353mode_mapping.fig'
    axes = torch.load(axes_path_mode_mapping)
    fig = torch.load(fig_path_mode_mapping)

    # Load models
    dyn_model = torch.load('./models/03_dyn_model/mode_disent1-20200607-1737dyn_model.pth')
    mode_model = torch.load('./models/04_mode_model/mode_disent1-20200608-0051mode_model.pth')

    # Environment
    env = OrdinaryEnvForPytorch('MountainCarContinuous-v0')

    tester = InteractiveDisentTester(
        dyn_model=dyn_model,
        mode_model=mode_model,
        device='cuda',
        env=env,
        mode_map_fig=fig,
        mode_map_axes=axes,
        num_episodes=1000,
        seed=0
    )

    tester.run()


if __name__ == '__main__':
    run()


