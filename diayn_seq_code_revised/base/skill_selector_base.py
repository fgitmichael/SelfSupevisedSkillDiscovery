import abc
import torch


class SkillSelectorBase(object):

    @abc.abstractmethod
    def get_random_skill(self) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    @property
    def skill_dim(self):
        raise NotImplementedError
