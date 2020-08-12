import torch
import gym
import numpy as np
from collections import deque
from typing import List, Union

from rlkit.samplers.data_collector.base import PathCollector
from rlkit.torch.sac.diayn.policies import MakeDeterministic

import self_supervised.utils.typed_dicts as td

from diayn_original_tb.seq_path_collector.rlkit_rollouter import Rollouter
from diayn_original_tb.policies.diayn_policy_extension import \
    SkillTanhGaussianPolicyExtension, MakeDeterministicExtension
from diayn_original_tb.seq_path_collector.rkit_seq_path_collector import SeqCollector

from diayn_no_oh.data_collector.rlkit_rollouter_no_oh import RollouterNoOH
from diayn_no_oh.policies.diayn_policy_no_oh import \
    SkillTanhGaussianPolicyNoOHTwoDim, MakeDeterministicExtensionNoOH


class SeqCollectorNoOH(SeqCollector):

    def __init__(self,
                 env: gym.Env,
                 policy: Union[
                     SkillTanhGaussianPolicyExtension,
                     SkillTanhGaussianPolicyNoOHTwoDim,
                     MakeDeterministicExtension,
                     MakeDeterministicExtensionNoOH],
                 max_num_epoch_paths_saved: int = None,
                 render: bool = False
                 ):
        super().__init__(
            env=env,
            policy=policy,
            max_num_epoch_paths_saved=max_num_epoch_paths_saved,
            render=render
        )

        # Overwrite base Rollouter
        self._rollouter = RollouterNoOH(
            env=env,
            policy=policy
        )

    def collect_new_paths(self,
                          seq_len: int,
                          num_seqs: int,
                          discard_incomplete_paths: bool=False
                          ):
        paths = self._collect_new_paths(
            seq_len=seq_len,
            num_seqs=num_seqs,
            discard_incomplete_paths=discard_incomplete_paths
        )

        # Extend TransitionModeMapping to TransitionModeMappingDiscreteSkills
        seq_dim = 1
        skill_id_seq = np.stack(
            [np.array([self._rollouter._policy.skill_id])] * seq_len,
            axis=seq_dim
        )
        assert skill_id_seq.shape[seq_dim] == paths[0].obs.shape[seq_dim]
        for (idx, path) in enumerate(paths):
            with_skill_id = td.TransitonModeMappingDiscreteSkills(
                **path,
                skill_id=skill_id_seq
            )
            paths[idx] = with_skill_id

        self._epoch_paths.extend(paths)
