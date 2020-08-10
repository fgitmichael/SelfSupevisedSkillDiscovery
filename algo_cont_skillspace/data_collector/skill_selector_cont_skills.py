import torch
import numpy as np
import random
from typing import Union, Tuple

from diayn_seq_code_revised.base.skill_selector_base import \
    SkillSelectorBase

import rlkit.torch.pytorch_util as ptu


class SkillSelectorContinous(SkillSelectorBase):

    def __init__(self, prior_skill_dist: torch.distributions.distribution.Distribution):
        assert len(prior_skill_dist.batch_shape) == 1
        self.skill_prior = prior_skill_dist

    def get_random_skill(self, batch_size=1) -> torch.Tensor:
        sampled = []
        for _ in range(batch_size):
            sampled.append(self.skill_prior.sample().to(ptu.device))

        sampled = torch.stack(sampled, dim=0)
        assert len(sampled.shape) == 2

        return sampled

    @property
    def skill_dim(self):
        return self.skill_prior.batch_shape[-1]




