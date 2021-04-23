import abc
import numpy as np

from latent_with_splitseqs.memory.replay_buffer_for_latent import LatentReplayBuffer

import self_supervised.utils.typed_dicts as td

from my_utils.np_utils.take_per_row import take_per_row
from my_utils.np_utils.np_array_equality import np_array_equality


class LatentReplayBufferSplitSeqSamplingBase(LatentReplayBuffer,
                                             metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _get_sample_seqlen(self) -> int:
        raise NotImplementedError

    @property
    def horizon_len(self):
        return self._seq_len

    def _extract_whole_batch(self, idx: tuple, **kwargs) -> td.TransitionModeMapping:
        assert 'seq_len' in kwargs.keys()
        seq_len = kwargs['seq_len']

        rows = idx[0]
        cols = idx[1]

        batch = td.TransitionModeMapping(
            obs=self._take_seqs(
                self._obs_seqs[rows],
                cols=cols,
                seq_len=seq_len,
            ),
            action=self._take_seqs(
                self._action_seqs[rows],
                cols=cols,
                seq_len=seq_len,
            ),
            reward=self._take_seqs(
                self._rewards_seqs[rows],
                cols=cols,
                seq_len=seq_len,
            ),
            next_obs=self._take_seqs(
                self._obs_next_seqs[rows],
                cols=cols,
                seq_len=seq_len,
            ),
            terminal=self._take_seqs(
                self._terminal_seqs[rows],
                cols=cols,
                seq_len=seq_len),
            mode=self._take_seqs(self._mode_per_seqs[rows], cols=cols, seq_len=seq_len),
        )

        return batch

    def _extract_batch_latent_training(self, idx, **kwargs):
        assert 'seq_len' in kwargs.keys()
        seq_len = kwargs['seq_len']

        rows = idx[0]
        cols = idx[1]

        batch = dict(
            next_obs=self._take_seqs(
                self._obs_next_seqs[rows],
                cols=cols,
                seq_len=seq_len,
            ),
            mode=self._take_seqs(
                self._mode_per_seqs[rows],
                cols=cols,
                seq_len=seq_len,
            ),
        )

        return batch

    def _take_seqs(self, array_: np.ndarray, cols, seq_len):
        """
        Return sampled seqs in (batch, data, seq) format (bds)
        """
        batch_dim = 0
        data_dim = 1
        seq_dim = 2
        array_bsd = np.swapaxes(array_, axis1=data_dim, axis2=seq_dim)
        seqs = take_per_row(array_bsd, cols, num_elem=seq_len)
        data_dim = 2
        seq_dim = 1
        return np.swapaxes(seqs, axis1=data_dim, axis2=seq_dim)

    def _sample_random_batch_extraction_idx(self, batch_size, **kwargs):
        assert 'seq_len' in kwargs.keys()
        seq_len = kwargs['seq_len']

        seqlen_cumsum = np.cumsum(
            self._seqlen_saved_paths[:self._size] - seq_len + 1
        )
        num_possible_idx = seqlen_cumsum[-1]
        sample_idx = np.random.randint(num_possible_idx, size=batch_size)
        rows = np.empty(batch_size, dtype=np.int)
        cols = np.empty(batch_size, dtype=np.int)
        for idx, sample_idx_ in enumerate(sample_idx):
            row = np.searchsorted(seqlen_cumsum, sample_idx_, side='right')
            col = sample_idx_ - seqlen_cumsum[row - 1] if row > 0 else sample_idx_
            rows[idx] = row
            cols[idx] = col

        max_cols = self._seqlen_saved_paths[rows]
        assert np.all((cols + seq_len) <= max_cols)

        #assert np.all(rows2 == rows)
        #assert np.all(cols2 == cols)

        return rows, cols

    def random_batch(self,
                     batch_size: int) -> td.TransitionModeMapping:
        """
        Sample batches with random sequence length
        Args:
            batch_size                 : N
        Return:
            TransitionModeMapping      : consisting of (N, data_dim, S) tensors
        """
        sample_seq_len = self._get_sample_seqlen()
        sample_idx = self._sample_random_batch_extraction_idx(
            batch_size,
            seq_len=sample_seq_len
        )
        batch_horizon = self._extract_whole_batch(sample_idx, seq_len=sample_seq_len)

        return td.TransitionModeMapping(
            **batch_horizon
        )

    def random_batch_latent_training(self,
                                     batch_size: int) -> dict:
        """
        Sample batches with random sequence length
        Args:
            batch_size              : N
        """
        sample_seq_len = self._get_sample_seqlen()
        sample_idx = self._sample_random_batch_extraction_idx(
            batch_size,
            seq_len=sample_seq_len
        )
        batch_horizon = self._extract_batch_latent_training(
            sample_idx,
            seq_len=sample_seq_len
        )

        return batch_horizon
