import torch
import numpy as np
import random
from typing import Union, Tuple

from diayn_seq_code_revised.base.skill_selector_base import \
    SkillSelectorBase

import rlkit.torch.pytorch_util as ptu

from diayn_seq_code_revised.networks.my_gaussian import MyGaussian
from diayn_no_oh.utils.hardcoded_grid_two_dim import NoohGridCreator


class SkillSelectorContinous(SkillSelectorBase):

    def __init__(self, prior_skill_dist: MyGaussian):
        self.skill_prior = prior_skill_dist
        self.grid = NoohGridCreator().get_grid()

    def get_random_skill(self, batch_size=1) -> torch.Tensor:
        dist = self.skill_prior(torch.tensor([1.]))
        sample =  dist.sample()
        assert sample.shape == torch.Size((self.skill_prior.output_size, ))

        return sample.to(ptu.device)

    @property
    def skill_dim(self):
        return self.skill_prior.output_size

    def get_skill_grid(self) -> torch.Tensor:
        return ptu.from_numpy(self.grid)