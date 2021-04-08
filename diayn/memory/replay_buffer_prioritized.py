import numpy as np

from diayn_cont.memory.replay_buffer import DIAYNContEnvReplayBuffer


class DIAYNEnvReplayBufferEBP(DIAYNContEnvReplayBuffer):

    def __init__(
            self,
            *args,
            calc_path_energy_fun,
            max_replay_buffer_size,
            **kwargs
    ):
        super().__init__(
            *args,
            max_replay_buffer_size=max_replay_buffer_size,
            **kwargs
        )
        self.calc_path_energy_fun = calc_path_energy_fun

        self.path_energy = {}
        self.path_energy['pot'] = np.zeros((max_replay_buffer_size,))
        self.path_energy['kin'] = np.zeros((max_replay_buffer_size,))
        self.path_energy['rot'] = np.zeros((max_replay_buffer_size,))

    def add_path(self, path):
        """
        Add a path to the replay buffer with calculation of path energy.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        # Calc path energy
        path_energy = self.calc_path_energy_fun(path)

        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
                path_energy=path_energy,
            )
        self.terminate_episode()

    def add_sample(
            self,
            *args,
            path_energy=None,
            **kwargs
    ):
        if path_energy is None:
            raise ValueError
        assert isinstance(path_energy, dict)
        assert "pot" in path_energy.keys() \
            and "kin" in path_energy.keys() \
            and "rot" in path_energy.keys()

        self.path_energy['pot'][self._top] = path_energy['pot']
        self.path_energy['kin'][self._top] = path_energy['kin']
        self.path_energy['rot'][self._top] = path_energy['rot']

        return super().add_sample(*args, **kwargs)

    def random_batch(self, batch_size):
        """
        Sample energy prioritized
        """
        pvals = {}
        pvals['pot'] = calc_pvals(self.path_energy['pot'][:len(self) - 1])
        pvals['kin'] = calc_pvals(self.path_energy['kin'][:len(self) - 1])
        pvals['rot'] = calc_pvals(self.path_energy['rot'][:len(self) - 1])

        pvals = 1/3 * pvals['pot'] + 1/3 * pvals['kin'] + 1/3 * pvals['rot']

        num_drawn_samples = np.random.multinomial(batch_size, pvals)
        indices = []
        for idx, num_drawn in enumerate(num_drawn_samples):
            for _ in range(num_drawn):
                indices.append(idx)
        indices = np.array(indices)

        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            skills=self._skill[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch


def calc_pvals(val_array: np.ndarray):
    assert len(val_array.shape) == 1

    # Calculate rankings
    # stackoverflow question:
    # "rank-items-in-an-array-using-python-numpy-without-sorting-array-twice"
    val_array = -val_array # descending
    sort_idx = val_array.argsort()
    ranks = np.empty_like(sort_idx)
    ranks[sort_idx] = np.arange(len(val_array))
    ranks = ranks + 1.
    ranks = ranks.astype(np.float)

    # Calculate pvals using  p(n) = 2n/(N(N+1))
    N = float(val_array.shape[0])
    factor = 2./(N*(N + 1))
    p_n = ranks * factor

    # Check
    assert np.sum(p_n) < 1.01 and np.sum(p_n) > 0.99

    return p_n
