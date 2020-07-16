from self_supervised.base.writer.writer_base import WriterDataMapping

class DiagnosticsWriter:

    def __init__(self,
                 log_interval: int,
                 ):
        self.log_interval = log_interval
        self._diagnostics = {}

    def write_diagnostic(self,
                         name: str,
                         data: WriterDataMapping):
        if name in self._diagnostics.keys():
            self._diagnostics[name].append(data)

        else:
            self._diagnostics[name] = [data]
