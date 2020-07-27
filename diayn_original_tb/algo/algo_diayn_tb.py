import torch
import torch.nn.functional as F
from typing import List
import gtimer as gt
import numpy as np


from rlkit.torch.sac.diayn.diayn_torch_online_rl_algorithm import \
    DIAYNTorchOnlineRLAlgorithm
import rlkit.torch.pytorch_util as ptu
from rlkit.core.rl_algorithm import _get_epoch_timings
from rlkit.core import logger

from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter

from self_sup_comb_discrete_skills.data_collector.path_collector_discrete_skills import \
    TransitonModeMappingDiscreteSkills

from diayn_original_tb.seq_path_collector.rkit_seq_path_collector import SeqCollector


class DIAYNTorchOnlineRLAlgorithmTb(DIAYNTorchOnlineRLAlgorithm):

    def __init__(self,
                 *args,
                 diagnostic_writer: DiagnosticsWriter,
                 seq_eval_collector: SeqCollector,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.diagnostic_writer = diagnostic_writer
        self.seq_eval_collector = seq_eval_collector

    def _end_epoch(self, epoch):
        super()._end_epoch(epoch)

        if self.diagnostic_writer.is_log(epoch):
            self.write_mode_influence(epoch)
            self.write_skill_hist(epoch)

    def write_mode_influence(self, epoch):
        paths = self._get_paths_mode_influence_test()

        obs_dim = paths[0].obs.shape[0]
        action_dim = paths[0].action.shape[0]
        for path in paths:
            self._write_mode_influence(
                path,
                obs_dim=obs_dim,
                action_dim=action_dim,
                epoch=epoch
            )

    def _write_mode_influence(self,
                              path,
                              obs_dim,
                              action_dim,
                              epoch
                              ):
        skill_id = path.skill_id.squeeze()[0]

        # Observations
        self.diagnostic_writer.writer.plot_lines(
            legend_str=["dim {}".format(i) for i in range(obs_dim)],
            tb_str="Mode Influence Test: Obs/Skill {}".format(skill_id),
            arrays_to_plot=path.obs,
            step=epoch,
            y_lim=[-3, 3]
        )

        # Actions
        self.diagnostic_writer.writer.plot_lines(
            legend_str=["dim {}".format(i) for i in range(action_dim)],
            tb_str="Mode Influence Test: Action/Skill {}".format(skill_id),
            arrays_to_plot=path.action,
            step=epoch,
            y_lim=[-1.2, 1.2]
        )

        # TODO: write rewards

    def _get_paths_mode_influence_test(self, seq_len=200) \
            -> List[TransitonModeMappingDiscreteSkills]:

        for skill in range(self.policy.skill_dim):
            # Set skill
            skill_oh = F.one_hot(
                ptu.tensor(skill), num_classes=self.policy.skill_dim)
            self.seq_eval_collector.set_skill(skill)


            self.seq_eval_collector.collect_new_paths(
                seq_len=seq_len,
                num_seqs=1,
                discard_incomplete_paths=False
            )

        mode_influence_eval_paths = self.seq_eval_collector.get_epoch_paths()

        return mode_influence_eval_paths

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )
        self.diagnostic_writer.writer.log_dict_scalars(
            dict_to_log=self.replay_buffer.get_diagnostics(),
            step=epoch,
            base_tag='Replay-Buffer Eval Stats'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')
        self.diagnostic_writer.writer.log_dict_scalars(
            dict_to_log=self.trainer.get_diagnostics(),
            step=epoch,
            base_tag='Trainer Eval Stats'
        )

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
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
        eval_paths = self.eval_data_collector.get_epoch_paths()
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

    def write_skill_hist(self, epoch):
        buffer_size = self.replay_buffer._size
        buffer_skills = np.argmax(self.replay_buffer._skill[:buffer_size], axis=-1)

        self.diagnostic_writer.writer.writer.add_histogram(
            tag="Buffer Skill Distribution",
            values=buffer_skills,
            global_step=epoch,
            bins=self.policy.skill_dim-1,
        )
