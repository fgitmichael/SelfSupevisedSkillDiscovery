import abc

from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter

from latent_with_splitseqs.base.evaluation_base import EvaluationBase


class MemoryEvalBase(EvaluationBase, metaclass=abc.ABCMeta):

    def __init__(self,
                 *args,
                 replay_buffer,
                 df_to_evaluate,
                 diagnostics_writer: DiagnosticsWriter,
                 **kwargs
                 ):
        super(MemoryEvalBase, self).__init__(*args, **kwargs)
        self.diagno_writer = diagnostics_writer
        self.replay_buffer = replay_buffer
        self.df_to_evaluate = df_to_evaluate

    def __call__(self, *args, epoch, **kwargs):
        memory_paths_dict = self.sample_paths_from_replay_buffer()
        df_ret_dict = self.apply_df(**memory_paths_dict)
        self.classifier_evaluation(epoch=epoch, **df_ret_dict, **memory_paths_dict)

    @abc.abstractmethod
    def sample_paths_from_replay_buffer(self):
        raise NotImplementedError
