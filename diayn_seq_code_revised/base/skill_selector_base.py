import abc
import torch


class SkillSelectorBase(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_random_skill(self) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def get_skill_grid(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def skill_dim(self):
        raise NotImplementedError
