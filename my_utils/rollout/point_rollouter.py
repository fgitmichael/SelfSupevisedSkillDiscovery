from typing import List, Union
import numpy as np

from my_utils.rollout.test_rollouter_base import TestRollouter


class PointRollouter(TestRollouter):

    def __init__(
            self,
            *args,
            skill_points: Union[List[np.ndarray], np.ndarray],
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        if isinstance(skill_points, list):
            assert isinstance(skill_points[0], np.ndarray)
            self._skills_to_rollout = np.stack(skill_points, axis=0)

        elif isinstance(skill_points, np.ndarray):
            assert len(skill_points.shape) == 2
            assert skill_points.shape[-1] == self.policy.skill
            self._skills_to_rollout = skill_points

        else:
            raise ValueError("Skills points must be list "
                             "of numpy arrays or numpy arrays")

    @property
    def skills_to_rollout(self) -> np.ndarray:
        return self._skills_to_rollout
