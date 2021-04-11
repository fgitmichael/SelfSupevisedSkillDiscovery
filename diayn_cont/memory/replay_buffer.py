import numpy as np
from gym.spaces import Discrete

from rlkit.torch.sac.diayn.diayn_env_replay_buffer import DIAYNEnvReplayBuffer


class DIAYNContEnvReplayBuffer(DIAYNEnvReplayBuffer):
    """
    Add eval function get_saved_skills
    Optimize path saving
    """

    def __len__(self):
        return self._size

    def get_saved_skills(self, unique=True) -> np.ndarray:
        batch_dim = 0
        if len(self) == self._max_replay_buffer_size:
            skills = self._skill
        elif len(self) < self._max_replay_buffer_size:
            assert len(self) == self._top
            skills = self._skill[:self._top]
        else:
            raise ValueError

        if unique:
            skills = np.unique(skills, axis=batch_dim)

        return skills

    def get_diagnostics(self):
        diagnostics = super().get_diagnostics()
        saved_skills = self.get_saved_skills()
        diagnostics['num_saved_skills'] = saved_skills.shape[0]
        return diagnostics

    def add_sample(
            self,
            *args,
            path_energy=None,
            **kwargs
    ):
        raise NotImplementedError("Adding samplings is handled fully in add path")

    def add_path(self, path):
        """
        Optimize path saving
        """
        path_len = path["observations"].shape[0]

        if isinstance(self._action_space, Discrete):
            new_actions = np.zeros((path_len, self._action_dim))
            new_actions[np.arange(path_len), path['actions']] = 1
        else:
            new_actions = path['actions']

        idx_to_write = np.mod(
            np.arange(self._top, self._top + path_len),
            self._max_replay_buffer_size
        )
        skills = np.stack([info["skill"] for info in path["agent_infos"]], axis=0)
        self._skill[idx_to_write] = skills
        self._actions[idx_to_write] = new_actions
        self._rewards[idx_to_write] = path["rewards"]
        self._observations[idx_to_write] = path["observations"]
        self._next_obs[idx_to_write] = path["next_observations"]
        self._terminals[idx_to_write] = path["terminals"]
        for key in self._env_info_keys:
            self._env_infos[key][idx_to_write] = path["env_infos"][key]

        for _ in range(path_len):
            self._advance()

        self.terminate_episode()
