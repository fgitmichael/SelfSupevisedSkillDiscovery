from self_supervised.utils.writer import WriterDataMapping, MyWriter


class DiagnosticsWriter:

    def __init__(self,
                 writer: MyWriter,
                 log_interval=None
                 ):
        self.log_interval = log_interval
        self._diagnostics = {}

        self.writer = writer

    def is_log(self, step, log_interval=None) -> bool:
        if log_interval is None:
            log_interval = self.log_interval

        if step % log_interval == 0:
            return True
