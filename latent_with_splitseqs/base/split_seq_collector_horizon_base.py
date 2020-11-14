import abc
import gym

from diayn_seq_code_revised.base.data_collector_base import DataCollectorRevisedBase
from diayn_seq_code_revised.base.rollouter_base import RolloutWrapperBase

class HorizonSplitSeqCollectorBase(DataCollectorRevisedBase, metaclass=abc.ABCMeta):

    def __init__(
            self,
            env: gym.Env,
            policy,
    ):
        self._rollouter = self.create_rollouter(
            env=env,
            policy=policy,
        )

    @abc.abstractmethod
    def create_rollouter(
            self,
            env,
            policy,
            **kwargs
    ) -> RolloutWrapperBase:
        raise NotImplementedError

    @abc.abstractmethod
    def collect_split_seq(
            self,
            seq_len,
            horizon_len,
            discard_if_incomplete
    ):
        raise NotImplementedError
