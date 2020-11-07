import warnings
from typing import Union

from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb

from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter

from rlkit.torch.torch_rl_algorithm import TorchTrainer


class NetParamHistogramLogger(object):

    def __init__(self,
                 trainer: TorchTrainer,
                 diagnostic_writer: DiagnosticsWriter):
        self.trainer = trainer
        self.diagnostic_writer = diagnostic_writer

    def __call__(self, *args, epoch, **kwargs):
        log_net_param_histograms(self, epoch)


def log_net_param_histograms(
        self: Union[
            DIAYNTorchOnlineRLAlgorithmTb,
            NetParamHistogramLogger
        ],
        epoch,
):
    assert isinstance(self.trainer, TorchTrainer)
    assert isinstance(self.diagnostic_writer, DiagnosticsWriter)

    for k, net in self.trainer.get_snapshot().items():
        for name, weight in net.named_parameters():
            try:
                self.diagnostic_writer.writer.writer. \
                    add_histogram(k + name, weight, epoch)
            except:
                warnings.warn("histogram didn't work")

            if weight.grad is not None:
                try:
                    self.diagnostic_writer.writer.writer. \
                        add_histogram(f'{k + name}.grad', weight.grad, epoch)
                except:
                    warnings.warn("histogram didn't work")
