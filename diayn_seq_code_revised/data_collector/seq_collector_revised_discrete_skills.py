import torch
import gym
import numpy as np
from collections import deque
from typing import List, Union

from rlkit.torch.sac.diayn.policies import MakeDeterministic

import self_supervised.utils.typed_dicts as td

from diayn_seq_code_revised.base.data_collector_base import PathCollectorRevisedBase
from diayn_seq_code_revised.policies.skill_policy import \
    SkillTanhGaussianPolicyRevised, MakeDeterministicRevised
from diayn_seq_code_revised.data_collector.rollouter_revised import RollouterRevised
from diayn_seq_code_revised.base.skill_selector_base import SkillSelectorBase
from diayn_seq_code_revised.data_collector.skill_selector import SkillSelectorDiscrete
from diayn_seq_code_revised.data_collector.seq_collector_revised import \
    SeqCollectorRevised

import rlkit.torch.pytorch_util as ptu

class SeqCollectorRevisedDiscreteSkills(SeqCollectorRevised):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._skill_id = None

    @property
    def skill(self):
        return self.policy.skill

    @skill.setter
    def skill(self, skill_with_id_dict):
        assert isinstance(skill_with_id_dict, dict)
        self.policy.skill = skill_with_id_dict['skill']
        self.skill_id = skill_with_id_dict['id']

    @property
    def skill_id(self):
        return self._skill_id

    @skill_id.setter
    def skill_id(self, skill_id: int):
        assert skill_id < self.skill_selector.num_skills
        self._skill_id = skill_id

    def skill_reset(self):
        random_skill, skill_id = self.skill_selector.get_random_skill(return_id=True)
        self.skill = dict(
            skill=random_skill,
            id=skill_id
        )

    def collect_new_paths(
            self,
            seq_len,
            num_seqs,
            obs_dim_to_select: Union[list, tuple] = None,
            **kwargs
    ):
        paths = self._collect_new_paths(
            seq_len=seq_len,
            num_seqs=num_seqs,
            obs_dim_to_select=obs_dim_to_select,
        )
        self._check_path(paths[0], seq_len)
        prepared_paths = self.prepare_paths_before_save(paths, seq_len)

        self._epoch_paths.extend(prepared_paths)

    def prepare_paths_before_save(self, paths, seq_len) \
            -> List[td.TransitionModeMapping]:
        # Extend to TransitionModeMappingDiscreteSkills
        seq_dim = 0
        skill_seq = np.stack(
            [ptu.get_numpy(self.skill)] * seq_len,
            axis=seq_dim
        )
        assert skill_seq.shape == (seq_len, self.skill_selector.skill_dim)
        skill_id = np.array([self.skill_id])
        skill_id_seq = np.stack([skill_id] * seq_len, axis=seq_dim)
        assert skill_id_seq.shape == (seq_len, 1)
        paths_with_skill_id = []
        for idx, path in enumerate(paths):
            with_skill_id = td.TransitonModeMappingDiscreteSkills(
                **path,
                mode=skill_seq,
                skill_id=skill_id_seq
            )
            paths_with_skill_id.append(with_skill_id)

        return paths_with_skill_id

    def get_epoch_paths(self, reset=True, transpose=True) -> List[td.TransitonModeMappingDiscreteSkills]:
        """
        Return:
            list of TransistionMapping consisting of (S, data_dim) np.ndarrays

        Only change to base: return typehint changed to discrete skills
        """
        assert len(self._epoch_paths) > 0

        epoch_paths = list(self._epoch_paths)

        if transpose:
            for idx, path in enumerate(epoch_paths):
                assert len(path.obs.shape) == 2
                epoch_paths[idx] = path.transpose(1, 0)

        if reset:
            self.reset()

        return epoch_paths

