import abc
from typing import Dict
import torch.nn as nn

from rlkit.core.trainer import Trainer

class MyTrainerBaseClass(Trainer, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train(self, data):
        pass

    @abc.abstractmethod
    def get_snapshot(self):
        return {}

    @abc.abstractmethod
    def end_epoch(self, epoch):
        pass

    @abc.abstractmethod
    def get_diagnostics(self):
        return {}

    @property
    @abc.abstractmethod
    def networks(self) -> Dict[str, nn.Module]:
        return {}

