import numpy as np
from typing import List, Type

import self_supervised.utils.typed_dicts as td

def split_path(
        split_seq_len: int,
        horizon_len: int,
        path_to_split: td.TransitionMapping,
) -> List[dict]:
    seq_dim = 0
    data_dim = 1

    assert path_to_split.obs.shape[seq_dim] == horizon_len
    assert horizon_len % split_seq_len == 0

    # Split paths into dict containing list of split obs, action, .. seqs
    split_dict_of_seqlists = {}
    num_chunks = horizon_len // split_seq_len
    for key, el in dict(path_to_split).items():
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

        split_seq_list.append(transition_mode_mapping_dict)

    return split_seq_list
