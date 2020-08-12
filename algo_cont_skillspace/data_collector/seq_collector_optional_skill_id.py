from typing import List, Union
import numpy as np

from diayn_seq_code_revised.data_collector.seq_collector_revised import SeqCollectorRevised

import self_supervised.utils.typed_dicts as td

import rlkit.torch.pytorch_util as ptu


class SeqCollectorRevisedOptionalSkillId(SeqCollectorRevised):

    def collect_new_paths(
            self,
            seq_len,
            num_seqs,
            skill_id=None,
            discard_incomplete_paths=None
    ):
        paths = self._collect_new_paths(
            num_seqs=num_seqs,
            seq_len=seq_len
        )

        if skill_id is not None:
            # Optionally extend to TransitionModeMappingDiscreteSkills
            assert type(skill_id) is int
            paths_to_save = self._extend_paths_with_skillid(
                seq_len,
                skill_id,
                paths
            )

        else:
            # Extend to TransitonModeMapping
            paths_to_save = self._extent_paths_with_skill(
                seq_len,
                paths
            )

        self._epoch_paths.extend(paths_to_save)

    def _extent_paths_with_skill(self, seq_len, paths) -> List[td.TransitionModeMapping]:
        # Extend to TransitionModeMapping
        seq_dim = 0
        skill_seq = np.stack(
            [ptu.get_numpy(self.skill)] * seq_len,
            axis=seq_dim
        )
        assert skill_seq.shape == (seq_len, self.skill_selector.skill_dim)

        paths_with_skills = []
        for (idx, path) in enumerate(paths):
            with_skill = td.TransitionModeMapping(
                **path,
                mode=skill_seq
            )
            paths_with_skills.append(with_skill)

        return paths_with_skills

    def _extend_paths_with_skillid(self,
                                   seq_len,
                                   skill_id,
                                   paths: List[td.TransitionMapping]):
        seq_dim = 0
        skill_seq = np.stack(
            [ptu.get_numpy(self.skill)] * seq_len,
            axis=seq_dim
        )
        assert skill_seq.shape == (seq_len, self.skill_selector.skill_dim)

        skill_id = np.array([skill_id])
        skill_id_seq = np.stack([skill_id] * seq_len,
                                axis=seq_dim)
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


