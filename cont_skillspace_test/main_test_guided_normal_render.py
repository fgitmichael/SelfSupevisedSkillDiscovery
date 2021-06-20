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
import argparse

from cont_skillspace_test.visualization_fun.env_viz_render import \
    EnvVisualizationRenderGuided
from cont_skillspace_test.utils.load_env import load_env

import rlkit.torch.pytorch_util as ptu


parser = argparse.ArgumentParser()
parser.add_argument('--epoch',
                    type=int,
                    default=100,
                    help="epoch to test",
                    )
parser.add_argument('--render_dt',
                    type=float,
                    default=0.00001,
                    help="render dt"
                    )
parser.add_argument('--horizon_len',
                    type=int,
                    default=500,
                    help="horizon length")
args = parser.parse_args()
ptu.set_gpu_mode(False)
epoch = args.epoch
extension = ".pkl"
policy_net_name = "policy_net_epoch{}".format(epoch) + extension
env = load_env()

policy = torch.load(policy_net_name, map_location=ptu.device)
env_viz = EnvVisualizationRenderGuided(
    env=env,
    policy=policy,
    seq_len=args.horizon_len,
    render_dt=args.render_dt,
)
env_viz.run()
