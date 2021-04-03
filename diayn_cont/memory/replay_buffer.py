import numpy as np

from rlkit.torch.sac.diayn.diayn_env_replay_buffer import DIAYNEnvReplayBuffer


class DIAYNContEnvReplayBuffer(DIAYNEnvReplayBuffer):

    def __len__(self):
        return self._size

    def get_saved_skills(self, unique=True) -> np.ndarray:
        batch_dim = 0
        if len(self) == self._max_replay_buffer_size:
            skills = self._skill
        elif len(self) < self._max_replay_buffer_size:
            assert len(self) == self._top
            skills = self._skill[:self._top, 0]
        else:
            raise ValueError

        if unique:
            skills = np.unique(skills, axis=batch_dim)

        return skills
