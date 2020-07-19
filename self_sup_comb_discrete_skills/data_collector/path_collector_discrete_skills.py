from typing import List
import torch

import numpy as np

from self_supervised.utils.typed_dicts import TransitonModeMappingDiscreteSkills
from self_supervised.base.data_collector.data_collector import \
    PathCollectorSelfSupervised


class  PathCollectorSelfSupervisedDiscreteSkills(PathCollectorSelfSupervised):

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.skill_id = None

    def set_discrete_skill(self,
                           skill_vec: torch.Tensor,
                           skill_id: int):
            super().set_skill(skill_vec)
            self.skill_id = skill_id

    def set_skill(self, skill: torch.Tensor):
        # To avoid signature change
        raise NotImplementedError('For this class use set discrete skills,'
                                  ' as skill id also has to be set')

    def collect_new_paths(
            self,
            seq_len: int,
            num_seqs: int,
            discard_incomplete_paths: bool = False,
    ):
        """
        Return:
            List of TransitionModeMappingDiscreteSkills
        """
        paths = super()._collect_new_paths(
            seq_len=seq_len,
            num_seqs=num_seqs,
            discard_incomplete_paths=discard_incomplete_paths
        )

        # Extend TransitonModeMapping to TransitionModeMappingDiscreteSkills
        seq_dim = 1
        skill_id_seq = np.stack(
            [np.array([self.skill_id])] * seq_len,
            axis=seq_dim
        )
        assert skill_id_seq.shape[seq_dim] == paths[0].obs.shape[seq_dim]
        for (idx, path) in enumerate(paths):
            with_skill_id = TransitonModeMappingDiscreteSkills(
                **path,
                skill_id=skill_id_seq
            )
            paths[idx] = with_skill_id

        self._epoch_paths.extend(paths)

    def get_epoch_paths(self) -> List[TransitonModeMappingDiscreteSkills]:
        """
        Return:
            list of TransistionMapping consisting of (S, dim) np.ndarrays
        """
        epoch_paths = list(self._epoch_paths)
        self.reset()

        return epoch_paths