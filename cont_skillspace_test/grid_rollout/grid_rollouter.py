import numpy as np
import tqdm
import math

from rlkit.samplers.rollout_functions import rollout as rollout_function
import rlkit.torch.pytorch_util as ptu

from cont_skillspace_test.grid_rollout.test_rollouter_base import TestRollouter


def create_twod_grid(
        low: np.ndarray,
        high: np.ndarray,
        num_points,
        **kwargs
) -> np.ndarray:
    """
    Creates grid with n=ceil(sqrt(num_points) points
    """
    assert low.shape == high.shape
    assert len(low.shape) == 1
    assert low.shape[0] == 2

    num_points_array = math.ceil(math.sqrt(num_points))
    mesh_grid_arrays = []
    for dim_low, dim_high in zip(low, high):
        array_ = np.linspace(dim_low, dim_high, num_points_array)
        mesh_grid_arrays.append(array_)

    grid_list = np.meshgrid(*mesh_grid_arrays)
    assert len(grid_list) == 2
    assert isinstance(grid_list, list)

    grid = np.stack(grid_list, axis=-1)

    return grid


class GridRollouter(TestRollouter):

    def __init__(self,
                 env,
                 policy,
                 horizon_len: int = 300,
                 ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.horizon_len = horizon_len

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

    def __call__(self, return_coverd_dists=False) -> list:
        assert self.skills_to_rollout is not None, \
            "No skill grid created yet, call create_skill_grid method!"

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
