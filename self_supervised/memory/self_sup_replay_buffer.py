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
            (max_replay_buffer_size, mode_dim)
        )

    def add_sample(self,
                   path: td.TransitionModeMapping,
                   **kwargs):
        """
        Args:
            path      : TransitionMapping consisting of (N, dim, S) np.ndarrays
            mode      : (N, mode_dim, S) np.ndarray
        """

        if mode is None:
            raise ValueError('Mode is needed')

        self._mode_per_seqs[self._top] = mode
        super().add_sample(
            path=path,
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
