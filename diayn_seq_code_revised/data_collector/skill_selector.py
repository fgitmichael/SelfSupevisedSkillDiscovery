import torch
import random

from diayn_seq_code_revised.base.skill_selector_base import \
    SkillSelectorBase

from diayn_no_oh.utils.hardcoded_grid_two_dim import get_oh_grid, get_no_oh_grid

class SkillSelectorDiscrete(SkillSelectorBase):

    def __init__(self, get_skill_grid_fun: [get_no_oh_grid,
                                            get_oh_grid]):
        self.skills = get_skill_grid_fun()

    def get_random_skill(self, return_id=False):
        randint = random.randint(0, self.skills.size(0) - 1)
        rand_skill = self.skills[randint]

        if return_id:
            return rand_skill, randint

        else:
            return rand_skill
