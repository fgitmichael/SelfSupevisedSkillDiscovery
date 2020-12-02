import abc

from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter


class PostEpochDiagnoWritingBase(object, metaclass=abc.ABCMeta):

    def __init__(
            self,
            *args,
            diagnostic_writer: DiagnosticsWriter,
            **kwargs
    ):
        self.diagno_writer = diagnostic_writer

    @abc.abstractmethod
    def __call__(self, *args, epoch, **kwargs):
        raise NotImplementedError
