# Working directory
import torch

from cont_skillspace_test.visualization_fun.env_viz_plot import \
    EnvVisualizationPlotHduvae

import rlkit.torch.pytorch_util as ptu


ptu.set_gpu_mode(False)

epoch = 40
extension = ".pkl"
policy_net_name = "policy_net_epoch{}".format(epoch) + extension
skill_selector_name = "skill_selector_epoch{}".format(epoch) + extension
env_name = "env" + extension

policy = torch.load(policy_net_name, map_location=ptu.device)
skill_selector = torch.load(skill_selector_name, map_location=ptu.device)
env = torch.load(env_name)

env_viz = EnvVisualizationPlotHduvae(
    env=env,
    policy=policy,
    skill_selector=skill_selector,
    seq_len=100,
    plot_offset=0.,
)
env_viz.run()
