import numpy as np
from typing import List

from diayn_seq_code_revised.data_collector.seq_collector_revised import \
    SeqCollectorRevised

import self_supervised.utils.typed_dicts as td


class SeqCollectorRevisedOptionalId(SeqCollectorRevised):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_id = None

    def collect_new_paths(
            self,
            seq_len,
            num_seqs,
            discard_incomplete_paths=None,
            id_to_add=None,
    ):
        if id_to_add is not None:
            self.skill_id = id_to_add

        super().collect_new_paths(
            seq_len=seq_len,
            num_seqs=num_seqs,
            discard_incomplete_paths=discard_incomplete_paths,
        )

    def prepare_paths_before_save(self, paths, seq_len) -> \
            List[td.TransitonModeMappingDiscreteSkills]:
        prepared_paths = super().prepare_paths_before_save(
            paths=paths,
            seq_len=seq_len
        )

        # Use self.skill_id and extend to TransitionModeMappingDiscrete as
        # this mapping provides skill ids
        seq_dim = 0
        skill_id_seq = np.stack(
            [np.array([self.skill_id])] * seq_len,
            axis=seq_dim
        )
        assert skill_id_seq.shape == (seq_len, 1)

        paths_with_id = []
        for (idx, path) in enumerate(prepared_paths):
            with_id = td.TransitonModeMappingDiscreteSkills(
                **path,
                skill_id=skill_id_seq
            )
            paths_with_id.append(with_id)

        return paths_with_id





