import torch
import numpy as np
import argparse

from cont_skillspace_test.grid_rollout.grid_rollout_test \
    import RolloutTesterPlot
from cont_skillspace_test.grid_rollout.grid_rollouter \
    import GridRollouter

import rlkit.torch.pytorch_util as ptu

parser = argparse.ArgumentParser()
parser.add_argument('--epoch',
                    type=int,
                    default=100,
                    help="epoch to test",
                    )
parser.add_argument('--num_eval_steps',
                    type=int,
                    default=1000,
                    help="number of rollout steps per io-selected skill",
                    )
parser.add_argument('--num_grid_points',
                    type=int,
                    default=200,
                    help="number of skill grid points")
args = parser.parse_args()

ptu.set_gpu_mode(False)
epoch = args.epoch
horizon_len = args.num_eval_steps

extension = ".pkl"
policy_net_name = "policy_net_epoch{}".format(epoch) + extension
config_name = "config" + extension
env_name = "env" + extension
env = torch.load(env_name)
policy = torch.load(policy_net_name, map_location=ptu.device)
config = torch.load(config_name)
assert config['skill_prior']['type'] == "uniform"
uniform_prior_low = config['skill_prior']['uniform']['low']
uniform_prior_high = config['skill_prior']['uniform']['high']
grid_rollouter = GridRollouter(
    env=env,
    policy=policy,
    horizon_len=horizon_len,
)
tester = RolloutTesterPlot(
    test_rollouter=grid_rollouter,
)
tester(
    grid_low=np.array([uniform_prior_low, uniform_prior_low]),
    grid_high=np.array([uniform_prior_high, uniform_prior_high]),
    num_points=args.num_grid_points,
)
