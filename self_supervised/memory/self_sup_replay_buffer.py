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
        self._mode_dim = mode_dim

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            seq_len=seq_len,
            env=env,
            env_info_sizes=env_info_sizes
        )

    def _create_memory(self):
        super()._create_memory()
        self._mode_per_seqs = np.zeros(
            (self._max_replay_buffer_size,
             self._mode_dim,
             self._seq_len),
            dtype=np.float32
        )

    @property
    def _objs_to_save(self):
        objs_to_save = super()._objs_to_save
        return dict(
            **objs_to_save,
            _mode_per_seqs=self._mode_per_seqs,
        )

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
        idx = self._sample_random_batch_extraction_idx(batch_size)
        batch = self._extract_whole_batch(idx)
        return batch

    def _sample_random_batch_extraction_idx(self, batch_size):
        return np.random.randint(0, self._size, batch_size)

    def _extract_whole_batch(self, idx: int) -> td.TransitionModeMapping:
        batch = td.TransitionModeMapping(
            obs=self._obs_seqs[idx],
            action=self._action_seqs[idx],
            reward=self._rewards_seqs[idx],
            next_obs=self._obs_next_seqs[idx],
            terminal=self._terminal_seqs[idx],
            mode=self._mode_per_seqs[idx]
        )
        return batch

    def get_diagnostics(self) -> dict:
        diagnostics_dict = super().get_diagnostics()
        batch_dim = 0
        saved_skills_unique = self.get_saved_skills(unique=True)
        diagnostics_dict['num_unique_skills'] = saved_skills_unique.shape[batch_dim]
        return diagnostics_dict

    def get_saved_skills(self, unique=True) -> np.ndarray:
        seq_dim = -1
        batch_dim = 0
        if len(self) == self._max_replay_buffer_size:
            skills = self._mode_per_seqs[..., 0]
        elif len(self) < self._max_replay_buffer_size:
            assert len(self) == self._top
            skills = self._mode_per_seqs[:self._top, ..., 0]
        else:
            raise ValueError

        if unique:
            skills = np.unique(skills, axis=batch_dim)

        return skills
