import numpy as np
import abc
import tqdm

import rlkit.torch.pytorch_util as ptu

from my_utils.grids.twod_grid import create_twod_grid
from my_utils.rollout.frame_plus_obs_rollout import rollout as rollout_function
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

    def rollout_trajectories(
            self,
            render=False,
            render_kwargs=None,
    ):
        if render and render_kwargs is None:
            render_kwargs = dict(
                mode='rgb_array'
            )

        rollouts = []
        for skill in tqdm.tqdm(self.skills_to_rollout):
            self.policy.skill = ptu.from_numpy(skill)
            rollout = rollout_function(
                env=self.env,
                agent=self.policy,
                max_path_length=self.horizon_len,
                render=True,
                render_kwargs=render_kwargs,
            )
            rollout['skill'] = ptu.get_numpy(self.policy.skill)
            rollouts.append(rollout)

        return rollouts
