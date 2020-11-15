import abc
import gym

from diayn_seq_code_revised.base.rollouter_base import RollouterBase

from rlkit.samplers.data_collector.base import PathCollector, DataCollector

class DataCollectorRevisedBase(DataCollector, metaclass=abc.ABCMeta):

    def __init__(
        self,
        env: gym.Env,
        policy,
    ):
        self._rollouter = self.create_rollouter(
            env=env,
            policy=policy,
        )
        self.reset()

    def end_epoch(self, epoch):
        super().end_epoch(epoch)
        self.reset()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def skill_reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def skill(self):
        raise NotImplementedError

    @abc.abstractmethod
    def create_rollouter(
            self,
            env,
            policy,
            **kwargs
    ) -> RollouterBase:
        raise NotImplementedError


class PathCollectorRevisedBase(DataCollectorRevisedBase, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def collect_new_paths(
            self,
            seq_len,
            num_seqs,
            **kwargs,
    ):
        raise NotImplementedError
