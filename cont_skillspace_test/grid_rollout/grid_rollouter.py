import numpy as np
import math

from rlkit.samplers.rollout_functions import rollout as rollout_function
import rlkit.torch.pytorch_util as ptu

from cont_skillspace_test.grid_rollout.test_rollouter_base import TestRollouter


class GridRollouter(TestRollouter):

    def __init__(self,
                 env,
                 policy,
                 horizon_len: int = 300,
                 ):
        self.env = env
        self.policy = policy
        self.horizon_len = horizon_len
        self.skills_to_rollout = None

    def create_skills_to_rollout(self,
                                 low: np.ndarray,
                                 high: np.ndarray,
                                 num_points: int,
                                 **kwargs
                                 ):
        assert low.shape == high.shape
        assert len(low.shape) == 1
        assert low.shape[0] == 2

        num_points_array = math.ceil(math.sqrt(num_points))
        mesh_grid_arrays = []
        for dim_low, dim_high in zip(low, high):
            array_ = np.meshgrid(dim_low, dim_high, num_points_array)
            mesh_grid_arrays.append(array_)

        grid_list = np.meshgrid(*mesh_grid_arrays)
        assert len(grid_list) == 2
        assert isinstance(grid_list, list)
        grid = np.concatenate([np.reshape(mat, (-1, 1)) for mat in grid_list], axis=1)

        self.skills_to_rollout = grid

    def __call__(self) -> list:
        assert self.skills_to_rollout is not None, \
            "No skill grid created yet, call create_skill_grid method!"

        rollouts = []
        for skill in self.skills_to_rollout:
            self.policy.skill = ptu.from_numpy(skill)
            rollout = rollout_function(
                env=self.env,
                agent=self.policy,
                max_path_length=self.horizon_len,
            )
            rollout['skill'] = ptu.get_numpy(self.policy.skill)
            rollouts.append(rollout)

        return rollouts
