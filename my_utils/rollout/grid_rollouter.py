import numpy as np
import abc
import tqdm

from my_utils.grids.twod_grid import create_twod_grid
from my_utils.rollout.test_rollouter_base import TestRollouter


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
