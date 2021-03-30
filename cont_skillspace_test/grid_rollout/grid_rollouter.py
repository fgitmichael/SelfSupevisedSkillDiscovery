import numpy as np
import tqdm
import math

from rlkit.samplers.rollout_functions import rollout as rollout_function
import rlkit.torch.pytorch_util as ptu

from cont_skillspace_test.grid_rollout.test_rollouter_base import TestRollouter


def create_twod_grid(
        low: np.ndarray,
        high: np.ndarray,
        num_points: int,
        matrix_form=False,
        **kwargs
) -> np.ndarray:
    """
    Creates grid with n=ceil(sqrt(num_points) points
    low             : (d,) np.array
    high            : (d,) np.array
    num_points      : N
    matrix_form     : bool
    """
    assert low.shape == high.shape
    assert len(low.shape) == 1

    num_points_array = math.ceil(math.sqrt(num_points))
    mesh_grid_arrays = []
    for dim_low, dim_high in zip(low, high):
        array_ = np.linspace(dim_low, dim_high, num_points_array)
        mesh_grid_arrays.append(array_)

    grid_list = np.meshgrid(*mesh_grid_arrays)
    assert len(grid_list) == low.shape[0]
    assert isinstance(grid_list, list)

    grid = np.stack(grid_list, axis=-1)
    if matrix_form:
        grid = np.reshape(grid, (num_points_array * num_points_array, low.shape[-1]))

    return grid


class GridRollouter(TestRollouter):

    def create_skills_to_rollout(self,
                                 *args,
                                 **kwargs
                                 ):
        self.skill_grid = create_twod_grid(*args, **kwargs)

    @property
    def skills_to_rollout(self) -> np.ndarray:
        assert self.skill_grid is not None
        return np.concatenate([np.reshape(mat, (-1, 1))
                               for mat in np.moveaxis(self.skill_grid, -1, 0)], axis=1)

    def rollout_trajectories(self):
        rollouts = []
        for skill in tqdm.tqdm(self.skills_to_rollout):
            self.policy.skill = ptu.from_numpy(skill)
            rollout = rollout_function(
                env=self.env,
                agent=self.policy,
                max_path_length=self.horizon_len,
            )
            rollout['skill'] = ptu.get_numpy(self.policy.skill)
            rollouts.append(rollout)

        return rollouts
