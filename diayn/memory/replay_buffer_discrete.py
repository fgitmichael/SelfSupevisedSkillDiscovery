import numpy as np

from diayn.memory.replay_buffer_opt import DIAYNEnvReplayBufferOpt


class DIAYNEnvReplayBufferOptDiscrete(DIAYNEnvReplayBufferOpt):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data_dim = -1
        self._num_times_skill_used_for_training = np.zeros((self._skill.shape[data_dim], ))

    @property
    def num_times_skill_used_for_training(self):
        return self._num_times_skill_used_for_training

    def _update_num_times_skill_used_for_training(self, skills):
        batch_dim = 0
        assert np.sum(skills) == skills.shape[batch_dim]
        self._num_times_skill_used_for_training += np.sum(skills, axis=batch_dim)

    def random_batch(self, batch_size) -> dict:
        batch = super().random_batch(batch_size)

        self._update_num_times_skill_used_for_training(batch['skills'])

        return batch