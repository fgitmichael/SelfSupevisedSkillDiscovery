import abc


from rlkit.samplers.data_collector.base import PathCollector

class PathCollectorRevisedBase(PathCollector):

    def end_epoch(self, epoch):
        super().end_epoch(epoch)
        self.reset()

    @abc.abstractmethod
    def reset(self):
        pass
