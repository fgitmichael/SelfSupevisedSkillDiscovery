import abc
import numpy as np

from latent_with_splitseqs.memory.replay_buffer_for_latent import LatentReplayBuffer

import self_supervised.utils.typed_dicts as td

from my_utils.np_utils.take_per_row import take_per_row
from my_utils.np_utils.np_array_equality import np_array_equality


class LatentReplayBufferSplitSeqSamplingBaseMemoryEfficient(
    LatentReplayBuffer,
    metaclass=abc.ABCMeta
):
    """
    Before: With terminal handling incomplete sequences take space of a complete sequence.
    MemoryEfficiency will be increased by saving observation sequences contiguous.
    """
    def __init__(self,
                 *args,
                 min_sample_seqlen: int = 2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._min_sample_seqlen = min_sample_seqlen
        self._single_mode = np.empty((self._max_replay_buffer_size, self._mode_dim))

    def _create_memory(self):
        self._mode_per_seqs = [None] * self._max_replay_buffer_size
        self._obs_seqs = [None] * self._max_replay_buffer_size
        self._obs_next_seqs = [None] * self._max_replay_buffer_size
        self._action_seqs = [None] * self._max_replay_buffer_size
        self._terminal_seqs = [None] * self._max_replay_buffer_size
        self._rewards_seqs = [None] * self._max_replay_buffer_size

    @abc.abstractmethod
    def _get_sample_seqlen(self) -> int:
        raise NotImplementedError

    @property
    def horizon_len(self):
        return self._seq_len

    def _extract_whole_batch(self, idx: tuple, **kwargs) -> td.TransitionModeMapping:
        """
        Return sampled seqs in (batch, data, seq) format (bds)
        """
        assert 'seq_len' in kwargs.keys()
        seq_len = kwargs['seq_len']

        rows = idx[0]
        cols = idx[1]
        batch_size = len(rows)

        obs = np.empty(
            (batch_size, self._observation_dim, seq_len),
            dtype=np.float32,
        )
        next_obs = np.empty(
            (batch_size, self._observation_dim, seq_len),
            dtype=np.float32,
        )
        actions = np.empty(
            (batch_size, self._action_dim, seq_len),
            dtype=np.float32,
        )
        terminal = np.empty(
            (batch_size, 1, seq_len),
            dtype=np.float32,
        )
        reward = np.empty(
            (batch_size, 1, seq_len),
            dtype=np.float32,
        )
        mode = np.empty(
            (batch_size, self._mode_dim, seq_len),
            dtype=np.float32,
        )
        for idx, (row, col) in enumerate(zip(rows, cols)):
            seq_dim = 1
            data_dim = 0
            horizon_len = self._obs_seqs[row].shape[seq_dim]
            if col > (horizon_len - seq_len) and self._padding:
                # Add Padding
                num_padding_els = col - (horizon_len - seq_len)
                col = col % seq_len

                zero_paddings_obs = np.zeros(
                    (self._obs_seqs[row].shape[data_dim], num_padding_els)
                )
                zero_paddings_action = np.zeros(
                    (self._action_seqs[row].shape[data_dim], num_padding_els)
                )
                paddings_rewards = np.stack(
                    [self._rewards_seqs[row][:, 0]] * num_padding_els,
                    axis=seq_dim,
                )
                paddings_mode = np.stack(
                    [self._mode_per_seqs[row][:, 0]] * num_padding_els,
                    axis=seq_dim,
                )
                padding_terminal = np.stack(
                    [self._terminal_seqs[row][:, 0]] * num_padding_els,
                    axis=seq_dim,
                )

                obs[idx] = np.concatenate(
                    [zero_paddings_obs, self._obs_seqs[row]],
                    axis=seq_dim,
                )[:, col:col + seq_len]
                next_obs[idx] = np.concatenate(
                    [zero_paddings_obs, self._obs_next_seqs[row]],
                    axis=seq_dim,
                )[:, col:col + seq_len]
                actions[idx] = np.concatenate(
                    [zero_paddings_action, self._action_seqs[row]],
                    axis=seq_dim,
                )[:, col:col + seq_len]
                reward[idx] = np.concatenate(
                    [paddings_rewards, self._rewards_seqs[row]],
                    axis=seq_dim,
                )[:, col:col + seq_len]
                terminal[idx] = np.concatenate(
                    [padding_terminal, self._terminal_seqs[row]],
                    axis=seq_dim,
                )[:, col:col + seq_len]
                mode[idx] = np.concatenate(
                    [paddings_mode, self._mode_per_seqs[row]],
                    axis=seq_dim,
                )[:, col:col + seq_len]

            elif col > (horizon_len - seq_len) and not self._padding:
                col = horizon_len - seq_len
                obs[idx] = self._obs_seqs[row][:, col:col + seq_len]
                next_obs[idx] = self._obs_next_seqs[row][:, col:col + seq_len]
                actions[idx] = self._action_seqs[row][:, col:col + seq_len]
                terminal[idx] = self._terminal_seqs[row][:, col:col + seq_len]
                reward[idx] = self._rewards_seqs[row][:, col:col + seq_len]
                mode[idx] = self._mode_per_seqs[row][:, col:col + seq_len]

            else:
                obs[idx] = self._obs_seqs[row][:, col:col + seq_len]
                next_obs[idx] = self._obs_next_seqs[row][:, col:col + seq_len]
                actions[idx] = self._action_seqs[row][:, col:col + seq_len]
                terminal[idx] = self._terminal_seqs[row][:, col:col + seq_len]
                reward[idx] = self._rewards_seqs[row][:, col:col + seq_len]
                mode[idx] = self._mode_per_seqs[row][:, col:col + seq_len]

        return td.TransitionModeMapping(
            obs=obs,
            action=actions,
            reward=reward,
            next_obs=next_obs,
            terminal=terminal,
            mode=mode,
        )

    def _extract_batch_latent_training(self, idx, **kwargs):
        assert 'seq_len' in kwargs.keys()
        seq_len = kwargs['seq_len']

        rows = idx[0]
        cols = idx[1]
        batch_size = len(rows)

        next_obs = np.empty(
            (batch_size, self._observation_dim, seq_len),
            dtype=np.float32,
        )
        mode = np.empty(
            (batch_size, self._mode_dim, seq_len),
            dtype=np.float32,
        )
        for idx, (row, col) in enumerate(zip(rows, cols)):
            seq_dim = 1
            data_dim = 0
            horizon_len = self._obs_seqs[row].shape[seq_dim]
            if col > (horizon_len - seq_len) and self._padding:
                # Add Padding
                num_padding_els = col - (horizon_len - seq_len)
                col = col % seq_len

                zero_paddings_obs = np.zeros(
                    (self._obs_seqs[row].shape[data_dim], num_padding_els)
                )
                paddings_mode = np.stack(
                    [self._mode_per_seqs[row][:, 0]] * num_padding_els,
                    axis=seq_dim,
                    )
                next_obs[idx] = np.concatenate(
                    [zero_paddings_obs, self._obs_next_seqs[row]],
                    axis=seq_dim,
                )[:, col:col + seq_len]
                mode[idx] = np.concatenate(
                    [paddings_mode, self._mode_per_seqs[row]],
                    axis=seq_dim,
                )[:, col:col + seq_len]

            elif col > (horizon_len - seq_len) and not self._padding:
                col = horizon_len - seq_len
                next_obs[idx] = self._obs_next_seqs[row][:, col:col + seq_len]
                mode[idx] = self._mode_per_seqs[row][:, col:col + seq_len]

            else:
                next_obs[idx] = self._obs_next_seqs[row][:, col:col + seq_len]
                mode[idx] = self._mode_per_seqs[row][:, col:col + seq_len]

        return dict(
            next_obs=next_obs,
            mode=mode,
        )

    def random_batch(self,
                     batch_size: int) -> td.TransitionModeMapping:
        sample_seq_len = self._get_sample_seqlen()
        sample_idx = self._sample_random_batch_extraction_idx(
            batch_size,
            seq_len=sample_seq_len
        )
        batch_horizon = self._extract_whole_batch(sample_idx, seq_len=sample_seq_len)

        return td.TransitionModeMapping(
            **batch_horizon
        )

    def random_batch_latent_training(self, batch_size: int) -> dict:
        sample_seq_len = self._get_sample_seqlen()
        sample_idx = self._sample_random_batch_extraction_idx(
            batch_size,
            seq_len=sample_seq_len,
        )
        batch_horizon = self._extract_batch_latent_training(
            sample_idx,
            seq_len=sample_seq_len,
        )

        return batch_horizon

    def add_sample(self,
                   path: td.TransitionModeMapping,
                   **kwargs):
        # Get path len
        seq_dim = 1
        assert len(path.obs.shape) == 2
        path_lens = [el.shape[seq_dim]
                     for el in path.values() if isinstance(el, np.ndarray)]
        assert all([path_len == path_lens[0] for path_len in path_lens])
        path_len = path_lens[0]

        # Only add complete paths
        if self._add_sample_if(path_len):
            self._seqlen_saved_paths[self._top] = path_len

            self._mode_per_seqs[self._top] = path.mode
            self._obs_seqs[self._top] = path.obs
            self._obs_next_seqs[self._top] = path.next_obs
            self._action_seqs[self._top] = path.action
            self._rewards_seqs[self._top] = path.reward
            self._terminal_seqs[self._top] = path.terminal
            self._single_mode[self._top] = path.mode[:, 0]

            self._advance()

    def _sample_random_batch_extraction_idx(self, batch_size, **kwargs):
        seqlen_cumsum_shortend = np.cumsum(
            self._seqlen_saved_paths[:self._size] - (self._min_sample_seqlen)
        )
        num_possible_idx = seqlen_cumsum_shortend[-1]
        sample_idx = np.random.randint(num_possible_idx, size=batch_size)
        rows = np.empty(batch_size, dtype=np.int)
        cols = np.empty(batch_size, dtype=np.int)
        for idx, sample_idx_ in enumerate(sample_idx):
            row = np.searchsorted(seqlen_cumsum_shortend, sample_idx_, side='right')
            col = sample_idx_ - seqlen_cumsum_shortend[row - 1] \
                if row > 0 else sample_idx_
            if col + self._min_sample_seqlen > self._seqlen_saved_paths[row]:
                raise ValueError
            rows[idx] = row
            cols[idx] = col

        max_cols = self._seqlen_saved_paths[rows]
        assert np.all(cols + self._min_sample_seqlen <= max_cols)

        return rows, cols

    def get_diagnostics(self) -> dict:
        diagnostics_dict = super().get_diagnostics()

        num_transitions = np.sum(self._seqlen_saved_paths[:self._size])
        average_path_lens = num_transitions/self._size
        median_path_lens = np.median(self._seqlen_saved_paths[:self._size])
        std_path_lens = np.std(self._seqlen_saved_paths[:self._size])
        min_path_len = np.min(self._seqlen_saved_paths[:self._size])
        max_path_len = np.max(self._seqlen_saved_paths[:self._size])

        diagnostics_dict['average_path_lens'] = average_path_lens
        diagnostics_dict['median_path_lens'] = median_path_lens
        diagnostics_dict['std_path_lens'] = std_path_lens
        diagnostics_dict['min_path_len'] = min_path_len
        diagnostics_dict['max_path_len'] = max_path_len
        num_transitions_key = 'num_transitions'
        assert num_transitions_key in diagnostics_dict.keys()
        diagnostics_dict[num_transitions_key] = num_transitions

        return diagnostics_dict

    def get_saved_skills(self, unique=True) -> np.ndarray:
        seq_dim = -1
        batch_dim = 0

        skills = self._single_mode

        if unique:
            skills = np.unique(skills, axis=batch_dim)

        return skills
