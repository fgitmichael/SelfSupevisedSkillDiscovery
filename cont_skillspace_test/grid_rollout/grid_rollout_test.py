import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

from my_utils.rollout.grid_rollouter import GridRollouterBase


class RolloutTesterPlot(object):

    def __init__(self,
                 test_rollouter: GridRollouterBase,
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

        # Plot in statespace
        subplot_int1 = 211
        ax1 = plt.gca()
        ax1.set_title('Statespace Plot')
        plt.subplot(subplot_int1)
        for idx, rollout in enumerate(grid_rollout):
            obs = rollout['observations']
            to_plot = [obs[:, 0], obs[:, 1]]
            if idx in max_sorted_idx:
                skill = rollout['skill']
                legend = "skill {}, covered distance {}".format(skill, obs[-1, 0])
                plt.plot(*to_plot, label=legend)
            else:
                plt.plot(*to_plot)
        plt.grid()
        plt.legend()

        # Plot colormap of covered distance
        subplot_int2 = 212
        plt.subplot(subplot_int2)
        coverd_dists = [rollout['observations'][-1, 0] for rollout in grid_rollout]
        coverd_dists_mat = np.reshape(
            np.expand_dims(np.array(coverd_dists), 0),
            self.test_rollouter.skill_grid.shape[:-1],
        )
        ax2 = plt.gca()
        ax2.set_title('Coverd Distance Colormap')
        plt.imshow(coverd_dists_mat)
        plt.colorbar(orientation='vertical')

        rollout_fig_name = 'epoch_' + str(epoch) + '.pdf'
        plt.savefig(os.path.join(
            self.path_name_grid_rollouts,
            rollout_fig_name
        ))

        if show:
            plt.show()
