from typing import List

import numpy as np

from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer
from self_supervised.utils import typed_dicts as td


class LatentReplayBuffer(SelfSupervisedEnvSequenceReplayBuffer):

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

        batch = dict(
            next_obs=self._obs_next_seqs[idx],
            mode=self._mode_per_seqs[idx],
        )

        return batch
