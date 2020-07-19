import numpy as np
from tqdm import tqdm
from typing import Dict, Union
import torch
import gtimer as gt
import matplotlib
from matplotlib import pyplot as plt

import self_supervised.utils.typed_dicts as td
from self_supervised.base.data_collector.data_collector import \
    PathCollectorSelfSupervised
from self_sup_comb_discrete_skills.data_collector.path_collector_discrete_skills import \
    PathCollectorSelfSupervisedDiscreteSkills
from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer
from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.base.algo.algo_base import BaseRLAlgorithmSelfSup
from self_supervised.base.writer.writer_base import WriterBase

import self_sup_combined.utils.typed_dicts as tdssc
from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter
from self_sup_combined.algo.trainer_sac import SelfSupCombSACTrainer
from self_sup_combined.algo.trainer_mode import ModeTrainer
from self_sup_combined.algo.algorithm import SelfSupCombAlgo

from self_sup_comb_discrete_skills.algo.mode_trainer_discrete_skill import \
    ModeTrainerWithDiagnosticsDiscrete
from self_sup_comb_discrete_skills.memory.replay_buffer_discrete_skills import \
    SelfSupervisedEnvSequenceReplayBufferDiscreteSkills
import self_sup_comb_discrete_skills.utils.typed_dicts as tdsscds

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger, eval_util
from rlkit.core.rl_algorithm import _get_epoch_timings

matplotlib.use('Agg')


class SelfSupCombAlgoDiscrete(SelfSupCombAlgo):

    def __init__(self,
                 sac_trainer: SelfSupCombSACTrainer,
                 mode_trainer: ModeTrainerWithDiagnosticsDiscrete,

                 exploration_env: NormalizedBoxEnvWrapper,
                 evaluation_env: NormalizedBoxEnvWrapper,

                 exploration_data_collector: PathCollectorSelfSupervisedDiscreteSkills,
                 evaluation_data_collector: PathCollectorSelfSupervisedDiscreteSkills,

                 replay_buffer: SelfSupervisedEnvSequenceReplayBufferDiscreteSkills,

                 diangnostic_writer: DiagnosticsWriter,
                 **kwargs
                 ):
        super().__init__(
            sac_trainer=sac_trainer,
            mode_trainer=mode_trainer,

            exploration_env=exploration_env,
            evaluation_env=evaluation_env,

            exploration_data_collector=exploration_data_collector,
            evaluation_data_collector=evaluation_data_collector,

            replay_buffer=replay_buffer,
            **kwargs
        )

        self.mode_dim = self.mode_trainer.model.mode_dim
        self.num_skills = self.mode_trainer.num_skills

        self.skill_idx_now = 0

        assert type(self.mode_trainer) == ModeTrainerWithDiagnosticsDiscrete
        self.discrete_skills = self.get_grid()

        self.diagnostic_writer = diangnostic_writer

    def _train_mode(self,
                    train_data: td.TransitonModeMappingDiscreteSkills
                    ):
        self.mode_trainer.train(
            data=tdsscds.ModeTrainerDataMappingDiscreteSkills(
                skills_gt=ptu.from_numpy(train_data.mode),
                obs_seq=ptu.from_numpy(train_data.obs),
                skill_id=ptu.from_numpy(train_data.skill_id)
            )
        )

    def set_next_skill(self,
                       path_collector: PathCollectorSelfSupervisedDiscreteSkills):
        assert type(path_collector) is PathCollectorSelfSupervisedDiscreteSkills

        skill_idx = np.random.randint(self.num_skills - 1)

        skill_vec = self.discrete_skills[skill_idx]

        path_collector.set_discrete_skill(
            skill_vec=skill_vec,
            skill_id=skill_idx,
        )

    def get_grid(self):
        assert type(self.mode_trainer) == ModeTrainerWithDiagnosticsDiscrete
        assert self.mode_trainer.num_skills == 10
        assert self.mode_trainer.model.mode_dim == 2

        # Hard coded for testing
        radius1 = 0.75
        radius2 = 1.
        radius3 = 1.38
        grid = np.array([
            [0., 0.],
            [radius1, 0.],
            [0., radius1],
            [-radius1, 0.],
            [0, -radius1],
            [radius2, radius2],
            [-radius2, radius2],
            [radius2, -radius2],
            [-radius2, -radius2],
            [0, radius3]
        ], dtype=np.float)

        grid = ptu.from_numpy(grid)

        return grid

    def _get_paths_mode_influence_test(self):
        assert type(self.eval_data_collector) is PathCollectorSelfSupervisedDiscreteSkills

        self.eval_data_collector.reset()
        for skill_id, discrete_skill in enumerate(self.discrete_skills):
            self.eval_data_collector.set_discrete_skill(
                skill_vec=discrete_skill,
                skill_id=skill_id
            )
            self.eval_data_collector.collect_new_paths(
                seq_len=self.seq_len,
                num_seqs=1,
            )

        mode_influence_eval_paths = self.eval_data_collector.get_epoch_paths()

        return mode_influence_eval_paths

    def write_mode_influence(self, epoch):
        paths = self._get_paths_mode_influence_test()

        obs_dim = self.policy.obs_dim
        action_dim = self.policy.action_dim
        for path in paths:
            assert path.obs.shape == (obs_dim, self.seq_len)
            assert path.action.shape == (action_dim, self.seq_len)

            skill_id = path.skill_id.squeeze()[0]

            self.diagnostic_writer.writer.plot_lines(
                legend_str=['dim' + str(i) for i in range(obs_dim)],
                tb_str="mode influence test: observations/mode {}".format(
                    skill_id),
                #arrays_to_plot=[dim for dim in obs],
                arrays_to_plot=path.obs,
                step=epoch
            )

            self.diagnostic_writer.writer.plot_lines(
                legend_str=["dim {}".format(dim) for dim in range(action_dim)],
                tb_str="mode influence test: actions/mode {}".format(
                    skill_id),
                arrays_to_plot=path.action,
                step=epoch
            )

            seq_dim = -1
            data_dim = 0
            path = path.transpose(seq_dim, data_dim)
            rewards = self.trainer.intrinsic_reward_calculator.mode_likely_based_rewards(
                obs_seq=ptu.from_numpy(path.obs).unsqueeze(dim=0),
                action_seq=ptu.from_numpy(path.action).unsqueeze(dim=0),
                skill_gt=ptu.from_numpy(path.mode).unsqueeze(dim=0)
            )
            assert rewards.shape == torch.Size((1, self.seq_len, 1))
            rewards = rewards.squeeze()
            assert rewards.shape == torch.Size((self.seq_len,))

            self.diagnostic_writer.writer.plot_lines(
                legend_str="skill_id {}".format(skill_id),
                tb_str="mode influence test rewards/skill_id {}".format(skill_id),
                arrays_to_plot=ptu.get_numpy(rewards),
                step=epoch
            )

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        gt.stamp('log outputting')

    def _end_epoch(self, epoch):
        super()._end_epoch(epoch)

        if self.diagnostic_writer.is_log(epoch):
            self.write_mode_influence(epoch)

        gt.stamp('saving')
        self._log_stats(epoch)
