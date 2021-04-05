import gtimer as gt

from diayn_original_tb.base.algo_base import SelfSupAlgoBase

from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter

from rlkit.core import logger, eval_util

class DIAYNContAlgo(SelfSupAlgoBase):

    def __init__(
            self,
            *args,
            diagnostic_writer: DiagnosticsWriter,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.diagnostic_writer = diagnostic_writer

    def _end_epoch(self, epoch):
        """
        Change order
        """
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch=epoch)
        self._log_stats(epoch)

        gt.stamp('end epoch')

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)
