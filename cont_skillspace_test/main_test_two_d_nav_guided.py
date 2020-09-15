import torch
import os
import argparse

from two_d_navigation_demo.env.navigation_env import TwoDimNavigationEnv
from cont_skillspace_test.visualization_fun.env_viz_plot import \
    EnvVisualizationPlotGuided

import rlkit.torch.pytorch_util as ptu

parser = argparse.ArgumentParser()
parser.add_argument('--epoch',
                    type=int,
                    default=100,
                    help="epoch to test",
                    )
args = parser.parse_args()

ptu.set_gpu_mode(True)
epoch = args.epoch
extension = ".pkl"
policy_net_name = "policy_net_epoch{}".format(epoch) + extension
env = TwoDimNavigationEnv()
policy = torch.load(policy_net_name)
env_viz = EnvVisualizationPlotGuided(
    env=env,
    policy=policy,
    seq_len=100,
)
env_viz.run()
