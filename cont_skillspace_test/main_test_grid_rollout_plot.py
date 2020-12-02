import torch
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
                    default=200,
                    help="number of rollout steps per io-selected skill",
                    )
args = parser.parse_args()

ptu.set_gpu_mode(False)
epoch = args.epoch
horizon_len = args.num_eval_steps

extension = ".pkl"
policy_net_name = "policy_net_epoch{}".format(epoch) + extension
env_name = "env" + extension
env = torch.load(env_name)
policy = torch.load(policy_net_name, map_location=ptu.device)
grid_rollouter = GridRollouter(
    env=env,
    policy=policy,
    horizon_len=horizon_len,
)
tester = RolloutTesterPlot(
    test_rollouter=grid_rollouter,
)
tester()

