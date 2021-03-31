import torch
import numpy as np
import argparse

from cont_skillspace_test.grid_rollout.grid_rollouter \
    import GridRollouter
from cont_skillspace_test.utils.load_env import load_env

from thes_graphics.rollout_relevant_eval.get_relevant_rollout_fun_using_identifier \
    import get_relevant_rollout_fun_using_identifier
from thes_graphics.relevant_trajectory_plotter.relevant_rollout_plotter \
    import RelevantTrajectoryPlotterSaver

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
parser.add_argument('--plot_width_inches_heat',
                    type=float,
                    default=3.7,
                    help="plot width (inches)")
parser.add_argument('--plot_height_inches_heat',
                    type=float,
                    default=3.7,
                    help="plot height (inches)")
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
                    nargs='+',
                    default=["max_abs_x"],
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
parser.add_argument('--heat_eval_fun',
                    type=str,
                    nargs='+',
                    default=['covered_dist'],
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

# Load relevant rollout extractor function
extract_relevant_rollout_fun = {
    id: get_relevant_rollout_fun_using_identifier(id)
    for id in args.extract_relevant_rollout_fun
}

# Load heat eval function
heat_eval_funs = {
    heat_eval_fun: get_heat_map_fun_using_identifier(heat_eval_fun)
    for heat_eval_fun in args.heat_eval_fun
}

# Load policy
epoch = args.epoch
policy_net_name = "policy_net_epoch{}".format(epoch) + extension
policy = torch.load(policy_net_name, map_location=ptu.device)

grid_rollouter = GridRollouter(
    env=env,
    policy=policy,
    horizon_len=horizon_len,
)

for id, relevant_extracting_fun in extract_relevant_rollout_fun.items():
    relevant_plotting = RelevantTrajectoryPlotterSaver(
        test_rollouter=grid_rollouter,
        plot_height_width_inches=(args.plot_height_inches, args.plot_width_inches),
        xy_label=(args.x_label, args.y_label),
        extract_relevant_rollouts_fun=relevant_extracting_fun,
        num_relevant_skills=args.num_relevant_skill,
        path=args.path,
        save_name_prefix= '02_' + args.filename + '_relevant_{}'.format(id),
    )
    relevant_plotting(
        epoch=epoch,
        grid_low=uniform_prior_edges[0],
        grid_high=uniform_prior_edges[1],
        num_points=args.num_grid_points,
    )

for heat_eval_fun_id, heat_eval_fun in heat_eval_funs.items():
    heatmap_plotting = HeatMapPlotterSaver(
        test_rollouter=grid_rollouter,
        plot_height_width_inches=(args.plot_height_inches, args.plot_width_inches),
        heat_eval_fun=heat_eval_fun,
        path=args.path,
        save_name_prefix='03_' + args.filename + '_heatmap_{}'.format(heat_eval_fun_id),
        uniform_skill_prior_edges=uniform_prior_edges,
        show=bool(args.show_plots),
    )
    heatmap_plotting(
        epoch=epoch,
        grid_low=uniform_prior_edges[0],
        grid_high=uniform_prior_edges[1],
        num_points=args.num_grid_points,
    )
