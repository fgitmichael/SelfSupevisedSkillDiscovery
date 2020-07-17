from typing import List

import numpy as np

from self_sup_comb_discrete_skills.memory.replay_buffer_discrete_skills import \
    TransitonModeMappingDiscreteSkills
from self_supervised.base.data_collector.data_collector import \
    PathCollectorSelfSupervised


class  PathCollectorSelfSupervisedDiscreteSkills(PathCollectorSelfSupervised):

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.skill_id = None

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
        skill_id_seq = np.stack([self.skill_id] * seq_len, dim=0)
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