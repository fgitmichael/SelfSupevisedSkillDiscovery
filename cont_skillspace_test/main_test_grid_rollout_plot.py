import torch
import numpy as np
import argparse
import pybulletgym

from cont_skillspace_test.grid_rollout.grid_rollout_test \
    import RolloutTesterPlot
from cont_skillspace_test.grid_rollout.grid_rollouter \
    import GridRollouter
from cont_skillspace_test.utils.load_env import load_env

import rlkit.torch.pytorch_util as ptu


parser = argparse.ArgumentParser()
parser.add_argument('--epoch',
                    type=int,
                    default=100,
                    help="epoch to test",
                    )
parser.add_argument('--grid_factor',
                    type=float,
                    default=None,
                    help="low, high of skills grid")
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

# Load policy
extension = ".pkl"
policy_net_name = "policy_net_epoch{}".format(epoch) + extension
policy = torch.load(policy_net_name, map_location=ptu.device)

# Load env
env = load_env()


if args.grid_factor is None:
    config_name = "config" + extension
    config = torch.load(config_name)
    assert config['skill_prior']['type'] == "uniform"
    uniform_prior_low = config['skill_prior']['uniform']['low']
    uniform_prior_high = config['skill_prior']['uniform']['high']

else:
    uniform_prior_low = -args.grid_factor
    uniform_prior_high = args.grid_factor

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
