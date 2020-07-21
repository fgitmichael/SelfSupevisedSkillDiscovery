import numpy as np
from typing import List
import gtimer as gt
import torch

from self_supervised.base.algo.algo_base import BaseRLAlgorithmSelfSup
from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
import self_supervised.utils.typed_dicts as td

from self_sup_combined.algo.algorithm import SelfSupCombAlgo

from self_sup_comb_discrete_skills.data_collector.path_collector_discrete_skills import \
    PathCollectorSelfSupervisedDiscreteSkills
from self_sup_comb_discrete_skills.memory.replay_buffer_discrete_skills import \
    SelfSupervisedEnvSequenceReplayBufferDiscreteSkills
from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter

from diayn_test.algo.diayn_trainer_seqwise import DiaynTrainerSeqwise

from rlkit.core import logger, eval_util
from rlkit.core.rl_algorithm import _get_epoch_timings
import rlkit.torch.pytorch_util as ptu

class DiaynAlgoSeqwise(BaseRLAlgorithmSelfSup):

    def __init__(self,
                 trainer: DiaynTrainerSeqwise,
                 exploration_env: NormalizedBoxEnvWrapper,
                 evaluation_env: NormalizedBoxEnvWrapper,
                 exploration_data_collector: PathCollectorSelfSupervisedDiscreteSkills,
                 evaluation_data_collector: PathCollectorSelfSupervisedDiscreteSkills,
                 replay_buffer: SelfSupervisedEnvSequenceReplayBufferDiscreteSkills,
                 batch_size,
                 seq_len: int,
                 num_epochs,
                 num_eval_steps_per_epoch,
                 num_expl_steps_per_train_loop,
                 num_trains_per_train_loop,
                 num_train_loops_per_epoch=1,
                 min_num_steps_before_training=0,
         ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

        self.seq_len = seq_len

        assert self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop, \
            'Online training presumes num_trains_per_train_loop ' \
            '>= num_expl_steps_per_train_loop'

        self.policy = self.trainer.policy

    def _train(self):
        self.training_mode(False)
        if self.min_num_steps_before_training > 0:
            self.set_next_skill(self.expl_data_collector)
            self.expl_data_collector.collect_new_paths(
                seq_len=self.seq_len,
                num_seqs=1,
                discard_incomplete_paths=False
            )
            init_expl_paths = self.expl_data_collector.get_epoch_paths()
            assert type(init_expl_paths[0]) == td.TransitonModeMappingDiscreteSkills
            self.replay_buffer.add_self_sup_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            gt.stamp('initial exploration', unique=True)

        num_trains_per_expl_step = self.num_trains_per_train_loop // \
            self.num_expl_steps_per_train_loop

        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs),
            save_itrs=True
        ):

            gt.stamp('evaluation sampling')

            self.set_next_skill(self.expl_data_collector)

            for _ in range(self.num_trains_per_train_loop):

                for _ in range(self.num_expl_steps_per_train_loop):

                    self.expl_data_collector.collect_new_paths(
                        seq_len=self.seq_len,
                        num_seqs=1,
                        discard_incomplete_paths=False
                    )
                    gt.stamp('exploration sampling', unique=False)

                    self.training_mode(True)
                    for _ in range(num_trains_per_expl_step):

                        train_data = self.replay_buffer.random_batch(
                            batch_size=self.batch_size
                        )
                        self.trainer.train(train_data)

                    self.training_mode(False)

            nex_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_self_sup_paths(nex_expl_paths)
            gt.stamp('data storing', unique=False)

            self._end_epoch(epoch)

    def set_next_skill(self,
                       path_collector: PathCollectorSelfSupervisedDiscreteSkills):
        skill_id = np.random.randint(
            low=0,
            high=10
        )
        skill_vec = self.trainer.skill_grid[skill_id]

        path_collector.set_discrete_skill(
            skill_vec=skill_vec,
            skill_id=skill_id
        )

    def training_mode(self, mode):
        for net in self.trainer.networks.values():
            net.train(mode)

    def to(self, device):
        for net in self.trainer.networks.values():
            net.to(device)

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


class DiaynAlgoSeqwiseTb(DiaynAlgoSeqwise):
    def __init__(self,
                 *args,
                 diangnostic_writer: DiagnosticsWriter,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.diagnostic_writer = diangnostic_writer

        self.skills = self.trainer.get_grid()

    def _end_epoch(self, epoch):
        super()._end_epoch(epoch)

        if self.diagnostic_writer.is_log(epoch):
            self.write_mode_influence(epoch)

    @torch.no_grad()
    def write_mode_influence(self, epoch):
        paths = self._get_paths_mode_influence_test()
        assert len(paths) == self.skills.size(0)

        obs_dim = self.policy.obs_dim
        action_dim = self.policy.action_dim

        for path in paths:
            assert path.obs.shape == (obs_dim, self.seq_len)
            assert path.action.shape == (action_dim, self.seq_len)
            assert path.skill_id.shape == (1, self.seq_len)
            assert np.stack([path.skill_id[:, 0]] * self.seq_len, axis=1) == path.skill_id

            skill_id = path.skill_id.squeeze()[0]

            # Observatons
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
                arrays_to_plot=path.obs,
                step=epoch,
                y_lim=[-3, 3]
            )

            #Rewards
            _, rewards = self.trainer.df_loss_rewards(
                skill_id=ptu.from_numpy(path.skill_id).long().unsqueeze(dim=0),
                next_obs=ptu.from_numpy(path.next_obs).unsqueeze(dim=0)
            )
            assert rewards.shape == torch.Size((1, self.seq_len, 1))
            rewards = rewards.squeeze()
            self.diagnostic_writer.writer.plot_lines(
                legend_str="rewards for skill {}".format(skill_id),
                tb_str="mode_influence test rewards/skill_id {}".format(skill_id),
                arrays_to_plot=ptu.get_numpy(rewards),
                step=epoch,
            )

    def _get_paths_mode_influence_test(self):
        self.eval_data_collector.reset()
        for skill_id, skill in enumerate(self.skills):
            self.eval_data_collector.set_discrete_skill(
                skill_vec=skill,
                skill_id=skill_id
            )
            self.eval_data_collector.collect_new_paths(
                seq_len=self.seq_len,
                num_seqs=1
            )

        mode_influence_eval_paths = self.eval_data_collector.get_epoch_paths()

        return mode_influence_eval_paths





