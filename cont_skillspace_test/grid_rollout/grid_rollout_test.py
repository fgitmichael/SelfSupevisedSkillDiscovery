import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

from cont_skillspace_test.grid_rollout.test_rollouter_base import TestRollouter


class RolloutTesterPlot(object):

    def __init__(self,
                 test_rollouter: TestRollouter,
                 ):
        self.test_rollouter = test_rollouter
        self.path_name_grid_rollouts = './grid_rollouts'
        if not os.path.exists(self.path_name_grid_rollouts):
            os.makedirs(self.path_name_grid_rollouts)

    def __call__(self,
                 *args,
                 epoch,
                 grid_low=np.array([-1.5, -1.5]),
                 grid_high=np.array([1.5, 1.5]),
                 num_points=200,
                 show=True,
                 **kwargs):
        # Rollout
        self.test_rollouter.create_skills_to_rollout(
            low=grid_low,
            high=grid_high,
            num_points=num_points,
        )
        grid_rollout = self.test_rollouter()

        # Find Rollout with big movements
        max_array = np.empty((len(grid_rollout)))
        for idx, rollout in enumerate(grid_rollout):
            obs = rollout['observations']
            max_ = np.amax(np.abs(obs[:, 0]))
            max_array[idx] = max_
        num_rollouts_with_legend = 10
        sorted_idx = np.argsort(max_array)
        max_sorted_idx = sorted_idx[::-1][:num_rollouts_with_legend]

        # Plot
        for idx, rollout in enumerate(grid_rollout):
            obs = rollout['observations']
            to_plot = [obs[:, 0], obs[:, 1]]
            if idx in max_sorted_idx:
                skill = rollout['skill']
                legend = "skill {}".format(skill)
                plt.plot(*to_plot, label=legend)
            else:
                plt.plot(*to_plot)
        plt.grid()
        plt.legend()
        rollout_fig_name = 'epoch_' + str(epoch) + '.pdf'
        plt.savefig(os.path.join(
            self.path_name_grid_rollouts,
            rollout_fig_name
        ))
        if show:
            plt.show()
