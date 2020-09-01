import torch
import os

from self_supervised.utils.writer import MyWriterWithActivation


class DiagnosticsWriter:

    def __init__(self,
                 writer: MyWriterWithActivation,
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

    def save_net(self,
                 net: torch.nn.Module,
                 save_name: str,
                 epoch: int,
                 log_interval: int
                 ):
        if self.is_log(epoch, log_interval) and epoch > 0:
            save_name = "{}_epoch{}".format(save_name, epoch)
            save_path = os.path.join(self.writer.model_dir, save_name)
            torch.save(net, save_path)
