from typing import Union
import gtimer as gt


from rlkit.torch.sac.diayn.diayn_torch_online_rl_algorithm import \
    DIAYNTorchOnlineRLAlgorithm
from rlkit.core.rl_algorithm import _get_epoch_timings
from rlkit.core import logger

from self_supervised.memory.self_sup_replay_buffer \
    import SelfSupervisedEnvSequenceReplayBuffer

from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter

from diayn_original_tb.seq_path_collector.rkit_seq_path_collector import SeqCollector

from diayn_seq_code_revised.data_collector.seq_collector_revised import \
    SeqCollectorRevised


class DIAYNTorchOnlineRLAlgorithmTb(DIAYNTorchOnlineRLAlgorithm):

    def __init__(self,
                 *args,
                 replay_buffer: SelfSupervisedEnvSequenceReplayBuffer,
                 diagnostic_writer: DiagnosticsWriter,
                 seq_eval_collector: Union[SeqCollector,
                                           SeqCollectorRevised],
                 mode_influence_one_plot_scatter=False,
                 mode_influence_paths_obs_lim: tuple=None,
                 **kwargs
                 ):
        super().__init__(
            *args,
            replay_buffer=replay_buffer,
            **kwargs
        )

        self.diagnostic_writer = diagnostic_writer
        self.seq_eval_collector = seq_eval_collector

        self.mode_influence_one_plot_scatter = mode_influence_one_plot_scatter
        self.mode_influence_path_obs_lim = mode_influence_paths_obs_lim

        self._epoch_cnt = 0
        self.post_epoch_funcs.append(self._update_epoch_cnt)

    def _update_epoch_cnt(self, epoch):
        self._epoch_cnt = epoch

    @property
    def epoch_cnt(self):
        return self._epoch_cnt

    def _end_epoch(self, epoch):
        """
        Change order compared to base method
        """
        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch=epoch)

        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving')
        self._log_stats(epoch)

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        #expl_paths = self.expl_data_collector.get_epoch_paths()
        #if hasattr(self.expl_env, 'get_diagnostics'):
        #    logger.record_dict(
        #        self.expl_env.get_diagnostics(expl_paths),
        #        prefix='exploration/',
        #    )
        #logger.record_dict(
        #    eval_util.get_generic_path_information(expl_paths),
        #    prefix="exploration/",
        #)
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        #eval_paths = self.eval_data_collector.get_epoch_paths()
        #if hasattr(self.eval_env, 'get_diagnostics'):
        #    logger.record_dict(
        #        self.eval_env.get_diagnostics(eval_paths),
        #        prefix='evaluation/',
        #    )
        #logger.record_dict(
        #    eval_util.get_generic_path_information(eval_paths),
        #    prefix="evaluation/",
        #)

        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
