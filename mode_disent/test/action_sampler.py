import torch

from abc import ABC, abstractmethod, ABCMeta


class ActionSampler(ABC):

    @abstractmethod
    def __call__(self, feature):
        """returns action"""


class ActionSamplerSeq(ActionSampler):

    def __init__(self, action_seq):
        assert len(action_seq.shape) == 2
        self.num_sequence = action_seq.size(0)
        self.action_seq = action_seq.detach().cpu()
        self._p = 0

    def _increment_p(self):
        self._p = (self._p + 1) % self.num_sequence

    def __call__(self, feature):
        action = self.action_seq[self._p].numpy()
        self._increment_p()
        return action
