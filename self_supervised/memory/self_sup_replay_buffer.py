from gym import Env
import numpy as np
from typing import List

from self_supervised.base.replay_buffer.env_replay_buffer import \
    SequenceEnvReplayBuffer
import self_supervised.utils.typed_dicts as td


class SelfSupervisedEnvSequenceReplayBuffer(SequenceEnvReplayBuffer):

    def __init__(self,
                 max_replay_buffer_size: int,
                 seq_len: int,
                 mode_dim: int,
                 env: Env,
                 env_info_sizes=None):
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            seq_len=seq_len,
            env=env,
            env_info_sizes=env_info_sizes
        )

        self._mode_per_seqs = np.zeros(
            (self._max_replay_buffer_size,
             mode_dim,
             self._seq_len),
            dtype=np.float32
        )

    def create_save_dict(self) -> dict:
        save_obj = super(SelfSupervisedEnvSequenceReplayBuffer, self).create_save_dict()
        save_obj['mode'] = self._mode_per_seqs
        return save_obj

    def process_save_dict(self, save_obj):
        self._mode_per_seqs = save_obj['mode']
        super().process_save_dict(save_obj)

    def add_sample(self,
                   path: td.TransitionModeMapping,
                   **kwargs):
        """
        Args:
            path      : TransitionMapping consisting of (N, dim, S) np.ndarrays
            mode      : (N, mode_dim, S) np.ndarray
        """
        self._mode_per_seqs[self._top] = path.pop('mode')

        super().add_sample(
            path=td.TransitionMapping(**path),
            **kwargs
        )

    def add_paths(self, paths: List[td.TransitionModeMapping]):
        # Avoid changing signature
        raise NotImplementedError("In this class add_self_sup_paths should be used!")

    def add_self_sup_paths(self,
                           paths: List[td.TransitionModeMapping]):
        """
        Args:
            paths           : TransitionMapping consiting of (N, 1, dim) arrays
        """
        for path in paths:
            self.add_sample(path)

    def random_batch(self,
                     batch_size: int) -> td.TransitionModeMapping:
        """
        Args:
            batch_size                 : N
        Return:
            TransitionModeMapping      : consisting of (N, data_dim, S) tensors
        """
        idx = np.random.randint(0, self._size, batch_size)

        batch = td.TransitionModeMapping(
            obs=self._obs_seqs[idx],
            action=self._action_seqs[idx],
            reward=self._rewards_seqs[idx],
            next_obs=self._obs_next_seqs[idx],
            terminal=self._terminal_seqs[idx],
            mode=self._mode_per_seqs[idx]
        )

        return batch
