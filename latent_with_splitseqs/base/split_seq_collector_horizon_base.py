import abc
import gym

from diayn_seq_code_revised.base.data_collector_base import DataCollectorRevisedBase

class HorizonSplitSeqCollectorBase(DataCollectorRevisedBase, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def collect_split_seq(
            self,
            seq_len,
            horizon_len,
            discard_if_incomplete
    ):
        raise NotImplementedError
