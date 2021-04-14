from typing import List

import numpy as np

from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer
from self_supervised.utils import typed_dicts as td


class LatentReplayBuffer(SelfSupervisedEnvSequenceReplayBuffer):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._seqlen_saved_paths = np.zeros((self._max_replay_buffer_size,))

    def add_paths(self, paths: List[td.TransitionModeMapping]):
        """
        To suppress pycharm warning
        """
        super().add_paths(paths)

    def random_batch_latent_training(self,
                                     batch_size: int) -> dict:
        """
        Sample only data relevant for training the latent model to save memory
        Args:
            batch_size      : N
        Returns:
            skill           : (N, skill_dim, S) nd-array
            next_obs        : (N, obs_dim, S) nd-array
        """
        idx = np.random.randint(0, self._size, batch_size)
        batch = self._extract_batch_latent_training(idx)
        return batch

    def _extract_batch_latent_training(self, idx):
        batch = dict(
            next_obs=self._obs_next_seqs[idx],
            mode=self._mode_per_seqs[idx],
        )
        return batch

    def get_diagnostics(self) -> dict:
        diagnostics_dict = super().get_diagnostics()

        average_path_lens = np.mean(self._seqlen_saved_paths)
        std_path_lens = np.std(self._seqlen_saved_paths)
        min_path_len = np.min(self._seqlen_saved_paths)
        max_path_len = np.max(self._seqlen_saved_paths)

        diagnostics_dict['average_path_lens'] = average_path_lens
        diagnostics_dict['std_path_lens'] = std_path_lens
        diagnostics_dict['min_path_len'] = min_path_len
        diagnostics_dict['max_path_len'] = max_path_len

        return diagnostics_dict

    def _add_sample_if(self, path_len) -> bool:
        return not path_len < self._seq_len

    def add_sample(self,
                   path: td.TransitionModeMapping,
                   **kwargs):
        # Get path len
        seq_dim = 1
        path_lens = [el.shape[seq_dim] for el in path.values() if isinstance(el, np.ndarray)]
        assert all([path_len == path_lens[0] for path_len in path_lens])
        path_len = path_lens[0]

        # Only add complete paths
        if self._add_sample_if(path_len):
            self._seqlen_saved_paths[self._top] = path_len

            self._mode_per_seqs[self._top, :, :path_len] = path.mode
            self._obs_seqs[self._top, :, :path_len] = path.obs
            self._obs_next_seqs[self._top, :, :path_len] = path.next_obs
            self._action_seqs[self._top, :, :path_len] = path.action
            self._rewards_seqs[self._top, :, :path_len] = path.reward
            self._terminal_seqs[self._top, :, :path_len] = path.terminal

            self._advance()
