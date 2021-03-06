import torch
import torch.nn.functional as F
from typing import List, Union
import numpy as np
import gtimer as gt


from rlkit.torch.sac.diayn.diayn_torch_online_rl_algorithm import \
    DIAYNTorchOnlineRLAlgorithm
import rlkit.torch.pytorch_util as ptu
from rlkit.core.rl_algorithm import _get_epoch_timings
from rlkit.core import logger

from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter

from diayn_original_tb.seq_path_collector.rkit_seq_path_collector import SeqCollector

from diayn_seq_code_revised.data_collector.seq_collector_revised import \
    SeqCollectorRevised

import self_supervised.utils.typed_dicts as td


class DIAYNTorchOnlineRLAlgorithmTb(DIAYNTorchOnlineRLAlgorithm):

    def __init__(self,
                 *args,
                 diagnostic_writer: DiagnosticsWriter,
                 seq_eval_collector: Union[SeqCollector,
                                           SeqCollectorRevised],
                 mode_influence_one_plot_scatter=False,
                 mode_influence_paths_obs_lim: tuple=None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.diagnostic_writer = diagnostic_writer
        self.seq_eval_collector = seq_eval_collector

        self.mode_influence_one_plot_scatter = mode_influence_one_plot_scatter
        self.mode_influence_path_obs_lim = mode_influence_paths_obs_lim

    def _end_epoch(self, epoch):
        super()._end_epoch(epoch)

        if self.diagnostic_writer.is_log(epoch):
            self.write_mode_influence_and_log(epoch)
            #self.write_skill_hist(epoch)
            gt.stamp('mode influence logging')

        self.diagnostic_writer.save_object(
            obj=self.policy,
            save_name="policy_net",
            epoch=epoch,
            log_interval=20,
        )

        if epoch == 0:
            self.diagnostic_writer.save_env(self.expl_env)

    def write_mode_influence_and_log(self, epoch):
        paths = self._get_paths_mode_influence_test()
        self._write_mode_influence_and_log(
            paths=paths,
            epoch=epoch,
        )

    def _write_mode_influence_and_log(self, paths, epoch):
        """
        Main logging function

        Args:
            eval_paths              : (data_dim, seq_dim) evaluation paths
                                      sampled directly
                                      from the environment
            epoch                   : int
        """
        obs_dim = paths[0].obs.shape[0]
        action_dim = paths[0].action.shape[0]

        # Plot influence in different plot
        for path in paths:
            self._write_mode_influence(
                path,
                obs_dim=obs_dim,
                action_dim=action_dim,
                epoch=epoch,
                obs_lim=self.mode_influence_path_obs_lim,
            )

        # Plot influence in one plot
        self._write_mode_influence_one_plot(
            paths=paths,
            epoch=epoch,
        )

    def _write_mode_influence_one_plot(self, paths, epoch):
        obs_dim = paths[0].obs.shape[0]
        action_dim = paths[0].action.shape[0]

        # Plot influence in one plot
        if obs_dim == 2:
            obs = np.stack([path.obs for path in paths], axis=2)
            self.diagnostic_writer.writer.plot(
                obs[0], obs[1],
                tb_str="ModeInfluence All Skills in One Plot/With Limits",
                step=epoch,
                labels=["skill {}".format(path.skill_id.squeeze()[0]) for path in paths],
                x_lim=[-2, 2],
                y_lim=[-2, 2]
            )

            obs = np.stack([path.obs for path in paths], axis=2)
            self.diagnostic_writer.writer.plot(
                obs[0], obs[1],
                tb_str="ModeInfluence All Skills in One Plot/Without Limits",
                step=epoch,
                labels=["skill {}".format(path.skill_id.squeeze()[0]) for path in paths],
            )

            if self.mode_influence_one_plot_scatter:
                self.diagnostic_writer.writer.scatter(
                    obs[0], obs[1],
                    tb_str="ModeInfluence All Skills in One Plot Scatter/With Limits",
                    step=epoch,
                    labels=["skill {}".format(path.skill_id.squeeze()[0]) for path in paths],
                    x_lim=[-2, 2],
                    y_lim=[-2, 2]
                )
                obs = np.stack([path.obs for path in paths], axis=2)
                self.diagnostic_writer.writer.scatter(
                    obs[0], obs[1],
                    tb_str="ModeInfluence All Skills in One Plot Scatter/Without Limits",
                    step=epoch,
                    labels=["skill {}".format(path.skill_id.squeeze()[0]) for path in paths],
                )

    def _write_mode_influence(self,
                              path,
                              obs_dim,
                              action_dim,
                              epoch,
                              obs_lim=None,
                              ):
        skill_id = path.skill_id.squeeze()[0]

        # Observations
        self.diagnostic_writer.writer.plot_lines(
            legend_str=["dim {}".format(i) for i in range(obs_dim)],
            tb_str="Mode Influence Test: Obs/Skill {}".format(skill_id),
            arrays_to_plot=path.obs,
            step=epoch,
            y_lim=[-3, 3] if obs_lim is None else obs_lim,
        )

        if obs_dim == 2:
            # State Space
            self.diagnostic_writer.writer.plot(
                *[obs_dim_array for obs_dim_array in path.obs],
                tb_str="State Space Behaviour/Skill {}".format(skill_id),
                step=epoch,
                x_lim=[-2.2, 2.2],
                y_lim=[-2.2, 2.2]
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

    def _get_paths_mode_influence_test(self, num_paths=1, seq_len=200) \
            -> List[td.TransitonModeMappingDiscreteSkills]:

        for _ in range(num_paths):
            for skill in range(self.policy.skill_dim):
                # Set skill
                skill_oh = F.one_hot(
                    ptu.tensor(skill), num_classes=self.policy.skill_dim)
                self.seq_eval_collector.set_skill(skill)

                self.seq_eval_collector.collect_new_paths(
                    seq_len=seq_len,
                    num_seqs=1,
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

    def write_skill_hist(self, epoch):
        buffer_size = self.replay_buffer._size
        buffer_skills = np.argmax(self.replay_buffer._skill[:buffer_size], axis=-1)

        self.diagnostic_writer.writer.writer.add_histogram(
            tag="Buffer Skill Distribution",
            values=buffer_skills,
            global_step=epoch,
            bins=self.policy.skill_dim-1,
        )
