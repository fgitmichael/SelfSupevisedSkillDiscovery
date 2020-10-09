from typing import List
import numpy as np
from warnings import warn

from diayn_seq_code_revised.data_collector.seq_collector_revised import SeqCollectorRevised

import self_supervised.utils.typed_dicts as td

import rlkit.torch.pytorch_util as ptu


class SeqCollectorSplitSeq(SeqCollectorRevised):

    def collect_new_paths(
            self,
            seq_len,
            num_seqs,
            horizon_len: int = None,
            discard_incomplete_paths=None,
            skill_id: int = None,

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
            )

            # Split paths
            split_paths = self._split_paths(
                split_seq_len=seq_len,
                horizon_len=horizon_len,
                paths_to_split=paths,
            )

            return split_paths

        else:
            # Do not split
            paths = super(SeqCollectorSplitSeq, self)._collect_new_paths(
                seq_len=seq_len,
                num_seqs=num_seqs,
            )

            return paths

    def _split_paths(self,
                     split_seq_len,
                     horizon_len,
                     paths_to_split: List[td.TransitionMapping]) \
            -> List[td.TransitionMapping]:
        return_paths = []

        for path in paths_to_split:
            return_paths.extend(
                self._split_path(
                    split_seq_len=split_seq_len,
                    horizon_len=horizon_len,
                    path_to_split=path)
            )

        return return_paths

    def _split_path(self,
                    split_seq_len: int,
                    horizon_len: int,
                    path_to_split: td.TransitionMapping) \
            -> List[td.TransitionMapping]:
        seq_dim = 0
        data_dim = 1

        assert path_to_split.obs.shape[seq_dim] == horizon_len
        assert horizon_len % split_seq_len == 0

        # Split paths into dict containing list of split obs, action, .. seqs
        split_dict_of_seqlists = {}
        num_chunks = horizon_len//split_seq_len
        for key, el in dict(path_to_split).items() :
            if isinstance(el, np.ndarray):
                split_dict_of_seqlists[key] = np.split(
                    el,
                    indices_or_sections=num_chunks,
                    axis=seq_dim,
                )

        # Convert dict of seq-lists into list of dicts
        split_seq_list = []
        for idx in range(num_chunks):

            transition_mode_mapping_dict = {}
            for key, seq_list in split_dict_of_seqlists.items():
                transition_mode_mapping_dict[key] = seq_list[idx]

            split_seq_list.append(
                td.TransitionMapping(
                    **transition_mode_mapping_dict,
                )
            )

        return split_seq_list
