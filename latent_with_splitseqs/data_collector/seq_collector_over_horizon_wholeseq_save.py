import numpy as np

from latent_with_splitseqs.base.seq_collector_over_horizon_base import SeqCollectorHorizonBase

import self_supervised.utils.typed_dicts as td


class SeqCollectorHorizonWholeSeqSaving(SeqCollectorHorizonBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._whole_seq_buffer = None

    def _save_split_seq(self, split_seq: td.TransitionModeMapping, horizon_completed: bool):
        assert isinstance(split_seq, td.TransitionModeMapping)
        seq_dim = 0
        data_dim = 1
        if self._whole_seq_buffer is None:
            self._whole_seq_buffer = split_seq
        else:
            new_buffer = {}
            for key, el_buffer in self._whole_seq_buffer.items():
                el_new = split_seq[key]
                new_buffer[key] = np.concatenate([el_buffer, el_new], axis=seq_dim)
            self._whole_seq_buffer = td.TransitionModeMapping(**new_buffer)

        if horizon_completed:
            self._epoch_split_seqs.append(self._whole_seq_buffer)
            self._whole_seq_buffer = None
