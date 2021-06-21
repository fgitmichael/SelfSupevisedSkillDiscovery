import numpy as np
import abc
import tqdm

from rlkit.samplers.rollout_functions import rollout as rollout_function
import rlkit.torch.pytorch_util as ptu

from cont_skillspace_test.grid_rollout.test_rollouter_base import TestRollouter
from my_utils.grids.twod_grid import create_twod_grid


class GridRollouterBase(TestRollouter, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_skills_to_rollout(self, **kwargs):
        raise NotImplementedError


class GridRollouter(GridRollouterBase):

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
