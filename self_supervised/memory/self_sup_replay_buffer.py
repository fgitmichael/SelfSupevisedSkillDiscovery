from gym import Env
import numpy as np
from typing import List

from self_supervised.base.replay_buffer.env_replay_buffer import \
    SequenceEnvReplayBuffer
from self_supervised.utils.typed_dicts import TransitionModeMapping


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
                   observation: np.ndarray,
                   action: np.ndarray,
                   reward: np.ndarray,
                   next_observation: np.ndarray,
                   terminal: np.ndarray,
                   mode: np.ndarray=None,
                   **kwargs):
        if mode is None:
            raise ValueError('Mode is needed')

        self._mode_per_seqs[self._top] = mode
        super().add_sample(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )

    def add_self_sup_path(self,
                          path: TransitionModeMapping):
        self.add_sample(
            observation=path.obs_seqs,
            action=path.action_seqs,
            reward=path.rewards,
            next_observation=path.next_obs_seqs,
            terminal=path.terminal_seqs,
            mode=path.mode,
        )

    def add_self_sup_paths(self,
                           paths: List[TransitionModeMapping]):
        for path in paths:
            self.add_self_sup_path(path)

    def random_batch(self,
                     batch_size: int) -> TransitionModeMapping:
        idx = np.random.randint(0, self._size, batch_size)

        batch = TransitionModeMapping(
            obs=self._obs_seqs[idx],
            action=self._action_seqs[idx],
            reward=self._rewards_seqs[idx],
            next_obs=self._obs_next_seqs[idx],
            terminal=self._terminal_seqs[idx],
            mode=self._mode_per_seqs[idx]
        )

        return batch
