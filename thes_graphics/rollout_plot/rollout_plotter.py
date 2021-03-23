import numpy as np
from typing import Tuple
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import matplotlib.pyplot as plt
import os
import tikzplotlib

from cont_skillspace_test.grid_rollout.test_rollouter_base import TestRollouter


class RolloutTesterPlotterThesGraphics(object):

    def __init__(self,
                 test_rollouter: TestRollouter,
                 extract_relevant_rollouts_fun,
                 num_relevant_skills: int,
                 path,
                 save_name_prefix,
                 plot_size_inches: float=None,
                 plot_height_width_inches: Tuple[float, float]=None,
                 xy_label: Tuple[str, str]=None,
                 ):
        self.test_rollouter = test_rollouter
        self.path_name_grid_rollouts = './grid_rollouts'

        self.extract_relevant_rollouts_fun = extract_relevant_rollouts_fun
        self.num_relevant_skills = num_relevant_skills

        self.path_name_grid_rollouts = path
        self.save_name_prefix = save_name_prefix

        assert plot_height_width_inches is not None \
               or plot_size_inches is not None
        self.plot_size_inches = plot_size_inches
        self.plot_height_width_inches = plot_height_width_inches
        self.plot_size_inches = plot_size_inches

        self.xy_label = xy_label \
            if xy_label is not None else ('', '')

        if not os.path.exists(self.path_name_grid_rollouts):
            os.makedirs(self.path_name_grid_rollouts)

    def __call__(self,
                 *args,
                 epoch,
                 grid_low=np.array([-1.5, -1.5]),
                 grid_high=np.array([1.5, 1.5]),
                 num_points=200,
                 **kwargs):
        # Rollout
        self.test_rollouter.create_skills_to_rollout(
            low=grid_low,
            high=grid_high,
            num_points=num_points,
        )
        grid_rollout = self.test_rollouter()

        # Extract relevant rollouts
        grid_rollout_relevant = self.extract_relevant_rollouts_fun(
            grid_rollout,
            num_to_extract=self.num_relevant_skills,
        )

        # Plot in statespace
        fig = plt.figure()
        ax1 = plt.gca()
        for idx, rollout in enumerate(grid_rollout_relevant):
            obs = rollout['observations']
            skill = rollout['skill']
            to_plot = [obs[:, 0], obs[:, 1]]
            legend = "skill {}".format(np.array2string(skill, separator=', '))
            plt.plot(*to_plot, label=legend)

        plt.grid()
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        self.set_fig_sizes(fig=fig)
        self.add_labels(ax=ax1)

        save_name_plot = os.path.join(
            self.path_name_grid_rollouts,
            self.save_name_prefix + '_trajectories.pgf'
        )
        save_name = os.path.join(self.path_name_grid_rollouts, save_name_plot)
        plt.savefig(save_name, bbox_inches='tight')

    def add_labels(self, ax: plt.Axes):
        ax.set_xlabel(self.xy_label[0])
        ax.set_ylabel(self.xy_label[1])

    def set_fig_sizes(self, fig: plt.Figure):
        if self.plot_size_inches is not None:
            fig.set_size_inches(w=self.plot_size_inches, h=self.plot_size_inches)
        elif self.plot_height_width_inches is not None:
            fig.set_figheight(self.plot_height_width_inches[0])
            fig.set_figwidth(self.plot_height_width_inches[1])
