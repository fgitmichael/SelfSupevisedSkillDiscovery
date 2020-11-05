import abc

from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter


class MemoryEvalBase(object, metaclass=abc.ABCMeta):

    def __init__(self,
                 replay_buffer,
                 df_to_evaluate,
                 diagnostics_writer: DiagnosticsWriter
                 ):
        self.diagno_writer = diagnostics_writer
        self.replay_buffer = replay_buffer
        self.df_to_evaluate = df_to_evaluate

    def __call__(self, epoch):
        memory_paths_dict = self.sample_paths_from_replay_buffer()
        df_ret_dict = self.apply_df(**memory_paths_dict)
        self.calc_classifier_performance(epoch, **df_ret_dict, **memory_paths_dict)

    @abc.abstractmethod
    def sample_paths_from_replay_buffer(self):
        raise NotImplementedError

    @abc.abstractmethod
    def apply_df(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def calc_classifier_performance(self, *args, **kwargs):
        raise NotImplementedError
