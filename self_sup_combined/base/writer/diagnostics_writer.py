from self_supervised.base.writer.writer_base import WriterDataMapping

from self_supervised.base.writer.writer_base import WriterBase

class DiagnosticsWriter:

    def __init__(self,
                 log_interval: int,
                 writer: WriterBase,
                 ):
        self.log_interval = log_interval
        self._diagnostics = {}

        self.writer = writer
