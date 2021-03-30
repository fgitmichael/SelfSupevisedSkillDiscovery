import torch
import numpy as np
import argparse

from cont_skillspace_test.grid_rollout.grid_rollouter \
    import GridRollouter
from cont_skillspace_test.utils.load_env import load_env

from thes_graphics.rollout_relevant_eval.get_relevant_rollout_fun_using_identifier \
    import get_relevant_rollout_fun_using_identifier
from thes_graphics.rollout_plot.rollout_plotter import RelevantTrajectoryPlotterSaver

import rlkit.torch.pytorch_util as ptu


parser = argparse.ArgumentParser()
parser.add_argument('--epoch',
                    type=int,
                    default=100,
                    nargs='+',
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
parser.add_argument('--plot_height_inches',
                    type=float,
                    default=3.7,
                    help="plot height (inches)")
parser.add_argument('--plot_width_inches',
                    type=float,
                    default=3.7,
                    help="plot width (inches)")
parser.add_argument('--x_label',
                    type=str,
                    default=None,
                    help="x label for plot")
parser.add_argument('--y_label',
                    type=str,
                    default=None,
                    help="y label for plot")
parser.add_argument('--extract_relevant_rollout_fun',
                    type=str,
                    default="max_abs_x",
                    help="identifier for the function "
                         "to extract the relevant rollout")
parser.add_argument('--num_relevant_skill',
                    type=int,
                    default=10,
                    help="number of relevant skills")
parser.add_argument('--path',
                    type=str,
                    default='./grid_rollouts',
                    help="path variable")
parser.add_argument('--filename',
                    type=str,
                    default='savedfig',
                    help="filename prefix")
args = parser.parse_args()

ptu.set_gpu_mode(False)
epochs = args.epoch
horizon_len = args.num_eval_steps

extension = ".pkl"

if args.grid_factor is None:
    config_name = "config" + extension
    config = torch.load(config_name)
    assert config['skill_prior']['type'] == "uniform"
    uniform_prior_low = config['skill_prior']['uniform']['low']
    uniform_prior_high = config['skill_prior']['uniform']['high']

else:
    uniform_prior_low = -args.grid_factor
    uniform_prior_high = args.grid_factor

# Skill prior tuple
uniform_prior_edges = (
    np.array([uniform_prior_low, uniform_prior_low]),
    np.array([uniform_prior_high, uniform_prior_high])
)

# Load env
env = load_env()

# Load relevant rollout extractor function
extract_relevant_rollout_fun = get_relevant_rollout_fun_using_identifier(
    args.extract_relevant_rollout_fun,
)

for epoch in epochs:
    # Load policy
    policy_net_name = "policy_net_epoch{}".format(epoch) + extension
    policy = torch.load(policy_net_name, map_location=ptu.device)

    grid_rollouter = GridRollouter(
        env=env,
        policy=policy,
        horizon_len=horizon_len,
    )
    tester = RelevantTrajectoryPlotterSaver(
        test_rollouter=grid_rollouter,
        plot_height_width_inches=(args.plot_height_inches, args.plot_width_inches),
        xy_label=(args.x_label, args.y_label),
        extract_relevant_rollouts_fun=extract_relevant_rollout_fun,
        num_relevant_skills=args.num_relevant_skill,
        path=args.path,
        save_name_prefix=args.filename,
    )
    tester(
        epoch=epoch,
        grid_low=uniform_prior_edges[0],
        grid_high=uniform_prior_edges[1],
        num_points=args.num_grid_points,
    )
