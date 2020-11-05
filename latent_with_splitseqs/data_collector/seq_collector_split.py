from typing import List, Union
import numpy as np
from warnings import warn

from diayn_seq_code_revised.data_collector.seq_collector_revised \
    import SeqCollectorRevised

import self_supervised.utils.typed_dicts as td

import rlkit.torch.pytorch_util as ptu

from latent_with_splitseqs.utils.split_path import split_path


class SeqCollectorSplitSeq(SeqCollectorRevised):

    def collect_new_paths(
            self,
            seq_len,
            num_seqs,
            horizon_len: int = None,
            discard_incomplete_paths=None,
            skill_id: int = None,
            obs_dim_to_select=None,

    ):
        paths = self._collect_new_paths(
            seq_len=seq_len,
            num_seqs=num_seqs,
            horizon_len=horizon_len
        )
        self._check_path(paths[0], seq_len)

        prepared_paths = self.prepare_paths_before_save(paths, seq_len)

        if skill_id is not None:
            paths_to_save = self._extend_with_skillid(
                seq_len=seq_len,
                skill_id=skill_id,
                paths=prepared_paths,
            )
            self._epoch_paths.extend(paths_to_save)

        else:
            self._epoch_paths.extend(prepared_paths)

    def _extend_with_skillid(self,
                             seq_len,
                             skill_id,
                             paths: List[td.TransitionModeMapping]) \
        -> List[td.TransitonModeMappingDiscreteSkills]:
        seq_dim = 0

        skill_id = np.array([skill_id])
        skill_id_seq = np.stack(
            [skill_id] * seq_len,
            axis=seq_dim,
        )
        assert skill_id_seq.shape == (seq_len, 1)

        paths_with_skill_id = []
        for idx, path in enumerate(paths):
            with_skill_id = td.TransitonModeMappingDiscreteSkills(
                **path,
                skill_id=skill_id_seq,
            )
            paths_with_skill_id.append(with_skill_id)

        return paths_with_skill_id

    def _collect_new_paths(
            self,
            seq_len,
            num_seqs,
            horizon_len: int = None,
            obs_dim_to_select: Union[list, tuple] = None,
            **kwargs,
    ):
        # Sanity check
        if horizon_len is not None:
            assert horizon_len >= seq_len
            if horizon_len % seq_len != 0:
                horizon_len = seq_len * (horizon_len//seq_len + 1)

            # Collect seqs
            paths = super(SeqCollectorSplitSeq, self)._collect_new_paths(
                seq_len=horizon_len,
                num_seqs=num_seqs,
                obs_dim_to_select=obs_dim_to_select,
            )

            # Split paths
            paths = self.split_paths(
                split_seq_len=seq_len,
                horizon_len=horizon_len,
                paths_to_split=paths,
            )

        else:
            # Do not split
            paths = super(SeqCollectorSplitSeq, self)._collect_new_paths(
                seq_len=seq_len,
                num_seqs=num_seqs,
                obs_dim_to_select=obs_dim_to_select,
            )

        return paths

    def split_paths(self,
                    split_seq_len,
                    horizon_len,
                    paths_to_split: List[td.TransitionMapping]) \
            -> List[td.TransitionMapping]:
        return_paths = []

        for path in paths_to_split:
            split_path_transition_mappings = self._split_path(
                split_seq_len=split_seq_len,
                horizon_len=horizon_len,
                path_to_split=path,
            )
            return_paths.extend(
                split_path_transition_mappings
            )

        return return_paths

    @staticmethod
    def _split_path(
            split_seq_len: int,
            horizon_len: int,
            path_to_split: td.TransitionMapping
    ) -> List[td.TransitionMapping]:
        split_path_dicts = split_path(
            split_seq_len=split_seq_len,
            horizon_len=horizon_len,
            path_to_split=path_to_split,
        )
        split_path_transition_mappings = [
            td.TransitionMapping(**split_path_dict)
            for split_path_dict in split_path_dicts
        ]

        return split_path_transition_mappings
