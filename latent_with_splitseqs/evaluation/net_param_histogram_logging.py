import warnings

from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb
from self_sup_combined.base.writer.is_log import is_log


@is_log
def _log_net_param_hist(self: DIAYNTorchOnlineRLAlgorithmTb, epoch):
    assert isinstance(self, DIAYNTorchOnlineRLAlgorithmTb)

    for k, net in self.trainer.network_dict.items():
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
