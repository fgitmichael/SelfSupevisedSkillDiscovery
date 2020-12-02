import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from cont_skillspace_test.grid_rollout.grid_rollouter \
    import GridRollouter

from cont_skillspace_test.grid_rollout.test_rollouter_base import TestRollouter


class RolloutTesterPlot(object):

    def __init__(self,
                 test_rollouter: TestRollouter,
                 ):
        self.test_rollouter = test_rollouter

    def __call__(self,
                 *args,
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

        # Plot
        for rollout in grid_rollout:
            obs = rollout['observations']
            plt.plot(obs[0], obs[1])

        plt.show()
