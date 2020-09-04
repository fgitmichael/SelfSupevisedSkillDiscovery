import torch
import os

from two_d_navigation_demo.env.navigation_env import TwoDimNavigationEnv
from cont_skillspace_test.rollout_fun.env_viz_two_d_nav import \
    EnvVisualizationTwoDNavGuidedasdfasfd

import rlkit.torch.pytorch_util as ptu

ptu.set_gpu_mode(True)
epoch = 20
extension = ".pkl"
policy_net_name = "policy_net_epoch{}".format(epoch) + extension
env = TwoDimNavigationEnv()
policy = torch.load(policy_net_name)
env_viz = EnvVisualizationTwoDNavGuidedasdfasfd(
    env=env,
    policy=policy,
    seq_len=100,
)
env_viz.run()
