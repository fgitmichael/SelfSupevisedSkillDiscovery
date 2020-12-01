# MountainCar path
#/home/michael/EIT/Github_Repos/24_SelfSupervisedDevel/seqwise_cont_skillspace/
# logsmountaincar/mode_disent0-20200905-1302 | seq_len: 70 | continous skill space
# | hidden rnn_dim: 20 | guided latent loss/model
import torch
import argparse

from cont_skillspace_test.visualization_fun.env_viz_plot import \
    EnvVisualizationPlotGuided

import rlkit.torch.pytorch_util as ptu

parser = argparse.ArgumentParser()
parser.add_argument('--epoch',
                    type=int,
                    default=100,
                    help="epoch to test",
                    )
parser.add_argument('--num_eval_steps',
                    type=int,
                    default=200,
                    help="number of rollout steps per io-selected skill",
                    )
args = parser.parse_args()

ptu.set_gpu_mode(False)
epoch = args.epoch
num_eval_steps = args.num_eval_steps

extension = ".pkl"
policy_net_name = "policy_net_epoch{}".format(epoch) + extension
env_name = "env" + extension
env = torch.load(env_name)
policy = torch.load(policy_net_name, map_location=ptu.device)
env_viz = EnvVisualizationPlotGuided(
    env=env,
    policy=policy,
    seq_len=num_eval_steps,
)
env_viz.run()
