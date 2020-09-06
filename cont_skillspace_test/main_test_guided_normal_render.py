# MountainCar path
#/home/michael/EIT/Github_Repos/24_SelfSupervisedDevel/seqwise_cont_skillspace/
# logsmountaincar/mode_disent0-20200905-1302 | seq_len: 70 | continous skill space
# | hidden rnn_dim: 20 | guided latent loss/model
# Cheetah path
#/home/michael/EIT/Github_Repos/24_SelfSupervisedDevel/
# seqwise_cont_skillspace/logshalfcheetah/
# mode_disent0-20200905-2312
# | seq_len: 70
# | continous skill space
# | hidden rnn_dim: 5
# | guided latent loss/model
import torch

from cont_skillspace_test.visualization_fun.env_viz_render import \
    EnvVisualizationRenderGuided

import rlkit.torch.pytorch_util as ptu

ptu.set_gpu_mode(False)
epoch = 40
extension = ".pkl"
policy_net_name = "policy_net_epoch{}".format(epoch) + extension
env_name = "env" + extension
env = torch.load(env_name)
policy = torch.load(policy_net_name, map_location=ptu.device)
env_viz = EnvVisualizationRenderGuided(
    env=env,
    policy=policy,
    seq_len=300,
    render_dt=0.002,
)
env_viz.run()
