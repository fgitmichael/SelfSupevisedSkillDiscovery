import abc
import numpy as np

from cont_skillspace_test.grid_rollout.test_rollouter_base import TestRollouter


class GridRolloutProcessor(object, metaclass=abc.ABCMeta):

    def __init__(self, test_rollouter: TestRollouter):
        self.test_rollouter = test_rollouter

    def __call__(self,
                 *args,
                 epoch,
                 grid_low=np.array([-1.5, -1.5]),
                 grid_high=np.array([1.5, 1.5]),
                 num_points=200,
                 **kwargs) -> list:
        # Rollout
        self.test_rollouter.create_skills_to_rollout(
            low=grid_low,
            high=grid_high,
            num_points=num_points,
        )
        grid_rollout = self.test_rollouter()

        return grid_rollout
