import abc

from rlkit.samplers.data_collector.base import StepCollector

class (StepCollector, metaclass=abc.ABCMeta):

    def end_epoch(self, epoch):
        self.reset()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

