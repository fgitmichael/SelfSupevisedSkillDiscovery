from typing import Union
import copy
import gtimer as gt


from rlkit.core.rl_algorithm import _get_epoch_timings
from rlkit.core import logger

from self_supervised.memory.self_sup_replay_buffer \
    import SelfSupervisedEnvSequenceReplayBuffer

from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter

from diayn_original_tb.seq_path_collector.rkit_seq_path_collector import SeqCollector
from diayn_original_tb.base.algo_base import SelfSupAlgoBase

from diayn_seq_code_revised.data_collector.seq_collector_revised import \
    SeqCollectorRevised

from latent_with_splitseqs.base.my_object_base import MyObjectBase


class DIAYNTorchOnlineRLAlgorithmTb(SelfSupAlgoBase, MyObjectBase):

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

        self._epoch_cnt = None

    @property
    def _objs_to_save(self):
        objs_to_save = super()._objs_to_save
        return dict(
            **objs_to_save,
            diagnostic_writer=self.diagnostic_writer,
            seq_eval_collector=self.seq_eval_collector,
        )

    def _end_epoch(self, epoch):
        """
        Add seq_eval collector
        """
        super()._end_epoch(epoch)
        self.seq_eval_collector.end_epoch(epoch)

    def train(self, start_epoch=0):
        super().train(start_epoch)

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
