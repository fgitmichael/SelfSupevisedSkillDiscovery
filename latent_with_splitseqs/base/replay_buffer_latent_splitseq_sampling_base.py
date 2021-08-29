import abc
import numpy as np

from latent_with_splitseqs.memory.replay_buffer_for_latent import LatentReplayBuffer

import self_supervised.utils.typed_dicts as td

from my_utils.np_utils.take_per_row import take_per_row
from my_utils.np_utils.np_array_equality import np_array_equality


class LatentReplayBufferSplitSeqSamplingBase(LatentReplayBuffer,
                                             metaclass=abc.ABCMeta):
    def __init__(
            self,
            *args,
            min_sample_seqlen: int = 2,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._min_sample_seqlen = min_sample_seqlen

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
                saved_seqlens=self._seqlen_saved_paths[rows],
                cols=cols,
                seq_len=seq_len,
            ),
            action=self._take_seqs(
                self._action_seqs[rows],
                saved_seqlens=self._seqlen_saved_paths[rows],
                cols=cols,
                seq_len=seq_len,
            ),
            reward=self._take_seqs(
                self._rewards_seqs[rows],
                saved_seqlens=self._seqlen_saved_paths[rows],
                cols=cols,
                seq_len=seq_len,
            ),
            next_obs=self._take_seqs(
                self._obs_next_seqs[rows],
                saved_seqlens=self._seqlen_saved_paths[rows],
                cols=cols,
                seq_len=seq_len,
            ),
            terminal=self._take_seqs(
                self._terminal_seqs[rows],
                saved_seqlens=self._seqlen_saved_paths[rows],
                cols=cols,
                seq_len=seq_len,
                padding_type='first_el',
            ),
            mode=self._take_seqs(
                self._mode_per_seqs[rows],
                saved_seqlens=self._seqlen_saved_paths[rows],
                cols=cols,
                seq_len=seq_len,
                padding_type='first_el',
            ),
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
                saved_seqlens=self._seqlen_saved_paths[rows],
                cols=cols,
                seq_len=seq_len,
            ),
            mode=self._take_seqs(
                self._mode_per_seqs[rows],
                saved_seqlens=self._seqlen_saved_paths[rows],
                cols=cols,
                seq_len=seq_len,
                padding_type='first_el'
            ),
        )

        return batch

    def _take_seqs(
            self,
            array_: np.ndarray,
            saved_seqlens: np.ndarray,
            cols,
            seq_len,
            padding_type=None
    ):
        """
        Return sampled seqs in (batch, data, seq) format (bds)
        """
        if padding_type is None:
            padding_type = 'zeros'

        batch_dim, data_dim, seq_dim = 0, 1, 2
        horizon_len = saved_seqlens
        assert array_.shape[batch_dim] == horizon_len.shape[0]

        # BSD Format
        array_bsd = np.swapaxes(array_, axis1=data_dim, axis2=seq_dim)
        batch_dim, data_dim, seq_dim = 0, 2, 1

        # Add padding
        end_idx = cols > horizon_len - seq_len
        if np.any(end_idx) and self._padding:
            num_padding_els = seq_len

            if padding_type == 'zeros':
                padding_array_ = np.zeros((
                    array_bsd.shape[batch_dim],
                    num_padding_els,
                    array_bsd.shape[data_dim]
                ))

            elif padding_type == 'first_el':
                padding_array_ = np.stack(
                    [array_bsd[:, 0, :]] * num_padding_els,
                    axis=seq_dim
                )

            else:
                raise NotImplementedError

            padded_array_ = np.concatenate([padding_array_, array_bsd], axis=seq_dim)

        elif np.any(end_idx) and not self._padding:
            cols[end_idx] = np.random.randint(self.horizon_len - seq_len, size=len(horizon_len[end_idx]))
            padded_array_ = array_bsd

        else:
            padded_array_ = array_bsd

        # Take seqs
        seqs = take_per_row(padded_array_, cols, num_elem=seq_len)

        return np.swapaxes(seqs, axis1=data_dim, axis2=seq_dim)

    def _sample_random_batch_extraction_idx(self, batch_size, **kwargs):
        assert 'seq_len' in kwargs.keys()
        seq_len = kwargs['seq_len']

        # If no terminal handling occured, sample normal way
        if np.sum(self._seqlen_saved_paths[:self._size]) < self._size * self.horizon_len:
            seqlen_cumsum = np.cumsum(
                self._seqlen_saved_paths[:self._size] # padding is needed
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

        else:
            rows = np.random.randint(0, self._size, size=batch_size)
            cols = np.random.randint(
                low=0,
                high=self.horizon_len, # padding is needed
                size=(batch_size,)
            )

        max_cols = self._seqlen_saved_paths[rows]
        assert np.all(cols <= max_cols)

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
