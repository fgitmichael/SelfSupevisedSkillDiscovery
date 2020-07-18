import numpy as np
from tqdm import tqdm
from typing import Dict, Union
import torch
import gtimer as gt
import matplotlib
from matplotlib import pyplot as plt

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

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger, eval_util
from rlkit.core.rl_algorithm import _get_epoch_timings

matplotlib.use('Agg')


class SelfSupCombAlgoDiscrete(SelfSupCombAlgo):

    def __init__(self,
                 *args,
                 mode_influence_diangnostic_writer: DiagnosticsWriter,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.mode_dim = self.mode_trainer.model.mode_dim
        self.num_skills = self.mode_trainer.num_skills

        self.skill_idx_now = 0

        assert type(self.mode_trainer) == ModeTrainerWithDiagnosticsDiscrete
        self.discrete_skills = self.get_grid()

        self.mode_influence_diagnostic_writer = mode_influence_diangnostic_writer

    def set_next_skill(self,
                       path_collector: PathCollectorSelfSupervisedDiscreteSkills):
        assert type(path_collector) is PathCollectorSelfSupervisedDiscreteSkills

        skill_idx = np.random.randint(self.num_skills - 1)
        path_collector.skill_id = skill_idx

        skill_vec = self.discrete_skills[skill_idx]

        path_collector.set_skill(skill_vec)

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

        self.eval_data_collector.reset()
        for discrete_skill in self.discrete_skills:
            self.eval_data_collector.set_skill(discrete_skill)
            self.eval_data_collector.collect_new_paths(
                seq_len=self.seq_len,
                num_seqs=1,
            )

        mode_influence_eval_paths = self.eval_data_collector.get_epoch_paths()

        return mode_influence_eval_paths

    def write_mode_influence(self, epoch):
        paths = self._get_paths_mode_influence_test()

        obs_dim = self.policy.obs_dim
        for path in paths:
            obs = path.obs
            assert obs.shape == (obs_dim, self.seq_len)

            self.mode_influence_diagnostic_writer.writer.plot_lines(
                legend_str=['dim' + str(i) for i in range(obs_dim)],
                tb_str='mode influence test',
                arrays_to_plot=[dim for dim in obs],
                step=epoch
            )

    def _end_epoch(self, epoch):
        super()._end_epoch(epoch)

        self.write_mode_influence(epoch)