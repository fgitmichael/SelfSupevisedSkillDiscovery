import abc


from rlkit.samplers.data_collector.base import PathCollector

class PathCollectorRevisedBase(PathCollector, metaclass=abc.ABCMeta):

    def end_epoch(self, epoch):
        super().end_epoch(epoch)
        self.reset()

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def collect_new_paths(
            self,
            seq_len,
            num_seqs,
            **kwargs,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def skill_reset(self):
        raise NotImplementedError
