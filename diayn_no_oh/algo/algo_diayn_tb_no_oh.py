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
from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb


class DIAYNTorchOnlineRLAlgorithmTbNoOH(DIAYNTorchOnlineRLAlgorithmTb):

    def _get_paths_mode_influence_test(self,
                                       seq_len=200) -> \
        List[TransitonModeMappingDiscreteSkills]:


        for skill_id in range(self.policy.num_skills):

            self.seq_eval_collector.set_skill(skill_id)
            self.seq_eval_collector.collect_new_paths(
                seq_len=seq_len,
                num_seqs=1,
                discard_incomplete_paths=False
            )

        mode_influence_eval_paths = self.seq_eval_collector.get_epoch_paths()

        return mode_influence_eval_paths
