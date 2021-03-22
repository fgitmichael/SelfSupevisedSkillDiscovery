import numpy as np
import math
from typing import Tuple
import matplotlib
import matplotlib.pyplot as plt
import os

from cont_skillspace_test.grid_rollout.test_rollouter_base import TestRollouter
from thes_graphics.heat_map_plot.plot_heat_map import plot_heat_map


class HeatMapPlotterSaver(object):

    def __init__(self,
                 test_rollouter: TestRollouter,
                 heat_eval_fun,
                 path,
                 save_name_prefix,
                 uniform_skill_prior_edges: tuple,
                 plot_size_inches: float=None,
                 plot_height_width_inches: Tuple[float, float]=None,
                 show=False,
                 ):
        self.test_rollouter = test_rollouter

        self.path_name_grid_rollouts = path
        self.save_name_prefix = save_name_prefix

        assert plot_height_width_inches is not None \
               or plot_size_inches is not None

        self.plot_size_inches = plot_size_inches
        self.plot_height_width_inches = plot_height_width_inches

        self.heat_eval_fun = heat_eval_fun
        self.uniform_skill_prior_edges = uniform_skill_prior_edges

        if not os.path.exists(self.path_name_grid_rollouts):
            os.makedirs(self.path_name_grid_rollouts)

        self.show_plot = show
        if not self.show_plot:
            matplotlib.use("pgf")
            matplotlib.rcParams.update({
                "pgf.texsystem": "pdflatex",
                'font.family': 'serif',
                'text.usetex': True,
                'pgf.rcfonts': False,
            })

    def __call__(self,
                 *args,
                 epoch,
                 grid_low: np.ndarray,
                 grid_high: np.ndarray,
                 num_points=200,
                 **kwargs):
        # Rollout
        self.test_rollouter.create_skills_to_rollout(
            low=grid_low,
            high=grid_high,
            num_points=num_points,
            matrix_form=False,
        )
        grid_rollout = self.test_rollouter()

        # Plot in statespace
        fig1 = plt.figure(1)
        ax1 = plt.gca()
        for idx, rollout in enumerate(grid_rollout):
            obs = rollout['observations']
            skill = rollout['skill']
            to_plot = [obs[:, 0], obs[:, 1]]
            legend = "skill {}".format(np.array2string(skill, separator=', '))
            plt.plot(*to_plot, label=legend)
        plt.grid()
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        self.set_fig_sizes(fig=fig1)

        # Calc heatmap
        heatmap_values_list = self.heat_eval_fun(
            grid_rollout=grid_rollout,
        )

        # Shape to equal row col array
        heatmap_values_arr = np.stack(heatmap_values_list, axis=0)
        n_rows = math.sqrt(heatmap_values_arr.shape[0])
        assert n_rows % 1 == 0
        n_rows = int(n_rows)
        heatmap_values_arr = np.reshape(heatmap_values_arr, (n_rows, n_rows))

        # Plot heatmap
        fig2 = plt.figure(2)
        plot_heat_map(
            fig=fig2,
            prior_skill_dist=self.uniform_skill_prior_edges,
            heat_values=heatmap_values_arr,
        )
        self.set_fig_sizes(fig=fig2)

        if self.show_plot:
            fig1.show()
            fig2.show()

        else:
            save_name_plot = os.path.join(
                self.path_name_grid_rollouts,
                self.save_name_prefix + '_trajectories.pgf'
            )
            save_name_heat = os.path.join(
                self.path_name_grid_rollouts,
                self.save_name_prefix + 'heatmap.pgf'
            )
            fig1.savefig(save_name_plot, bbox_inches='tight')
            fig2.savefig(save_name_heat, bbox_inches='tight')

    def set_fig_sizes(self, fig: plt.Figure):
        if self.plot_size_inches is not None:
            fig.set_size_inches(w=self.plot_size_inches, h=self.plot_size_inches)
        elif self.plot_height_width_inches is not None:
            fig.set_figheight(self.plot_height_width_inches[0])
            fig.set_figwidth(self.plot_height_width_inches[1])