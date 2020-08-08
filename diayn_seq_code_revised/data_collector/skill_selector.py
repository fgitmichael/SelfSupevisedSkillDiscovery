import torch
import numpy as np
import random
from typing import Union, Tuple

from diayn_seq_code_revised.base.skill_selector_base import \
    SkillSelectorBase

import rlkit.torch.pytorch_util as ptu


class SkillSelectorDiscrete(SkillSelectorBase):

    def __init__(self, get_skill_grid_fun):
        self.skills = get_skill_grid_fun()

    def get_random_skill(
            self,
            return_id=False) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        randint = random.randint(0, self.skills.shape[0] - 1)
        rand_skill = ptu.from_numpy(self.skills[randint])

        if return_id:
            return rand_skill, randint
        else:
            return rand_skill

    def contains(self, skill: torch.Tensor):
        skill_np = ptu.get_numpy(skill)

        assert len(skill) == 1
        assert skill.shape[-1] == self.skills.shape[-1]

        dists = np.linalg.norm(self.skills - skill_np, axis=-1)
        idx = np.nonzero(dists < 0.001)[0]

        if idx.shape == (1,):
            return True
        else:
            return False

    @property
    def skill_dim(self):
        return self.skills.shape[-1]

    @property
    def num_skills(self):
        return self.skills.shape[0]

    def get_skill_grid(self) -> torch.Tensor:
        """
        In the discrete cas simple return the skill matrix
        """
        return ptu.from_numpy(self.skills)
