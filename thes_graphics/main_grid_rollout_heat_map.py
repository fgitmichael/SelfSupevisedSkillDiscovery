import torch
import numpy as np
import argparse
import pybulletgym

from cont_skillspace_test.utils.load_env import load_env
from cont_skillspace_test.grid_rollout.grid_rollouter \
    import GridRollouter

import rlkit.torch.pytorch_util as ptu

from thes_graphics.heat_map_eval.get_heat_map_fun_using_identifier \
    import get_heat_map_fun_using_identifier
from thes_graphics.heat_map_plot.heat_map_plot_saver import HeatMapPlotterSaver


parser = argparse.ArgumentParser()
parser.add_argument('--epoch',
                    type=int,
                    default=100,
                    help="epoch to test",
                    )
parser.add_argument('--num_eval_steps',
                    type=int,
                    default=100,
                    help="number of rollout steps per io-selected skill",
                    )
parser.add_argument('--num_grid_points',
                    type=int,
                    default=9,
                    help="number of skill grid points")
parser.add_argument('--plot_height_inches',
                    type=float,
                    default=3.7,
                    help="plot height (inches)")
parser.add_argument('--plot_width_inches',
                    type=float,
                    default=3.7,
                    help="plot width (inches)")
parser.add_argument('--path',
                    type=str,
                    default='./grid_rollouts',
                    help="path variable")
parser.add_argument('--filename',
                    type=str,
                    default='saved_fig',
                    help="filename prefix")
parser.add_argument('--heat_eval_fun',
                    type=str,
                    default='covered_dist',
                    help="heat eval function identifier"
                    )
parser.add_argument('--show_plots',
                    type=int,
                    default=0,
                    help="heat eval function identifier",
                    )

args = parser.parse_args()

ptu.set_gpu_mode(False)
epochs = args.epoch
horizon_len = args.num_eval_steps

extension = ".pkl"

# Skill prior tuple
config_name = "config" + extension
config = torch.load(config_name)
assert config['skill_prior']['type'] == "uniform"
uniform_prior_low = config['skill_prior']['uniform']['low']
uniform_prior_high = config['skill_prior']['uniform']['high']

uniform_prior_edges = (np.array([uniform_prior_low, uniform_prior_low]),
                       np.array([uniform_prior_high, uniform_prior_high]))

# Load env
env = load_env()

# Load heat eval function
heat_eval_fun = get_heat_map_fun_using_identifier(args.heat_eval_fun)

# Load policy
epoch = args.epoch
policy_net_name = "policy_net_epoch{}".format(epoch) + extension
policy = torch.load(policy_net_name, map_location=ptu.device)

grid_rollouter = GridRollouter(
    env=env,
    policy=policy,
    horizon_len=horizon_len,
)
tester = HeatMapPlotterSaver(
    test_rollouter=grid_rollouter,
    plot_height_width_inches=(args.plot_height_inches, args.plot_width_inches),
    heat_eval_fun=heat_eval_fun,
    path=args.path,
    save_name_prefix=args.filename,
    uniform_skill_prior_edges=uniform_prior_edges,
    show=bool(args.show_plots),
)
tester(
    epoch=epoch,
    grid_low=uniform_prior_edges[0],
    grid_high=uniform_prior_edges[1],
    num_points=args.num_grid_points,
)
