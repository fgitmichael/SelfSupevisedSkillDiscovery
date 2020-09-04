# Working directory
# /home/michael/EIT/Github_Repos/24_SelfSupervisedDevel/
# seqwise_cont_highdimusingvae/logs2dnav/
# mode_disent0-20200904-1549 | seq_len: 100 | continous
# skill space | hidden rnn_dim: 20 | step training repeat: 1/model
import torch

from two_d_navigation_demo.env.navigation_env import TwoDimNavigationEnv
from cont_skillspace_test.rollout_fun.env_viz_plot import \
    EnvVisualizationPlotHduvae

import rlkit.torch.pytorch_util as ptu

ptu.set_gpu_mode(True)
epoch = 20
extension = ".pkl"
policy_net_name = "policy_net_epoch{}".format(epoch) + extension
skill_selector_name = "skill_selector_epoch{}".format(epoch) + extension
env = TwoDimNavigationEnv()
policy = torch.load(policy_net_name)
skill_selector = torch.load(skill_selector_name)
env_viz = EnvVisualizationPlotHduvae(
    env=env,
    policy=policy,
    skill_selector=skill_selector,
    seq_len=100,
)
env_viz.run()