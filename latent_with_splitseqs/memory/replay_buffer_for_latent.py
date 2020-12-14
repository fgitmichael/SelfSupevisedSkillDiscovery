import numpy as np

from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer


class LatentReplayBuffer(SelfSupervisedEnvSequenceReplayBuffer):

    def random_batch_latent_training(self,
                                     batch_size: int) -> dict:
        """
        Sample only data relevant for training the latent model to save memory

        Returns:
            skill           : (N, skill_dim, S) nd-array
            next_obs        : (N, obs_dim, S) nd-array
        """
        idx = np.random.randint(0, self._size, batch_size)

        batch = dict(
            next_observations=self._obs_next_seqs[idx],
            skills=self._mode_per_seqs[idx],
        )

        return batch
