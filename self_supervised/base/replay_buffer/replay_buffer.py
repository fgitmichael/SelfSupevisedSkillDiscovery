import numpy as np
from prodict import Prodict
from collections import OrderedDict

from self_supervised.base.replay_buffer.replay_buffer_base import SequenceReplayBuffer

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer


class SequenceBatch(Prodict):
    obs_seqs: np.ndarray
    action_seqs: np.ndarray
    rewards: np.ndarray
    terminal_seqs: np.ndarray
    next_obs_seqs: np.ndarray

    def __init__(self,
                 obs_seqs: np.ndarray,
                 action_seqs: np.ndarray,
                 reward_seqs: np.ndarray,
                 terminal_seqs: np.ndarray,
                 next_obs_seqs: np.ndarray):

        super(SequenceBatch, self).__init__(
            obs_seqs=obs_seqs,
            action_seqs=action_seqs,
            reward_seqs=reward_seqs,
            terminal_seqs=terminal_seqs,
            next_obs_seqs=next_obs_seqs
        )


class NormalSequenceReplayBuffer(SequenceReplayBuffer):

    def __init__(self,
                 max_replay_buffer_size,
                 observation_dim,
                 action_dim,
                 seq_len,
                 env_info_sizes=None):

        if env_info_sizes is None:
            env_info_sizes = dict()

        self._observation_dim = observation_dim
        self._action_dim = action_dim

        self._max_replay_buffer_size = max_replay_buffer_size
        self._seq_len = seq_len

        self._obs_seqs = np.zeros(
            (self._max_replay_buffer_size,
             self._observation_dim,
             self._seq_len)
        )
        self._obs_next_seqs = np.zeros(
            (self._max_replay_buffer_size,
             self._observation_dim,
             self._seq_len)
        )
        self._action_seqs = np.zeros(
            (self._max_replay_buffer_size,
             self._action_dim,
             self._seq_len)
        )
        self._rewards_seqs = np.zeros(
            (self._max_replay_buffer_size,
             1,
             self._seq_len)
        )
        self._terminal_seqs = np.zeros(
            (self._seq_len,
             1,
             self._max_replay_buffer_size),
            dtype='uint8'
        )

        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros(
                (self._max_replay_buffer_size,
                 size,
                 self._seq_len)
            )
        self._env_info_keys = env_info_sizes.keys()

        self._top = 0
        self._size = 0

    def add_sample(self,
                   observation: np.ndarray,
                   action: np.ndarray,
                   reward: np.ndarray,
                   next_observation: np.ndarray,
                   terminal: np.ndarray,
                   **kwargs):
        """
        Args:
            observation         : (obs_dim, path_len) array of observations
            action              : (action_dim, path_len) array of actions
            reward              : (1, path_len) of rewards
            next_observation    : (obs_dim, path_len) array of observations
            terminal            : (1, path_len) of uint8's
        """
        self._test_dimensions(
            observation = observation,
            action = action,
            reward = reward,
            next_observation = next_observation,
            terminal = terminal)

        self._obs_seqs[self._top] = observation
        self._action_seqs[self._top] = action
        self._obs_next_seqs[self._top] = next_observation
        self._rewards_seqs[self._top] = reward
        self._terminal_seqs[self._top] = terminal

        self._advance()

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size

        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def terminate_episode(self):
        pass

    def random_batch(self, batch_size: int) -> SequenceBatch:
        idx = np.random.randint(
            low=0,
            high=self._size,
            size=batch_size
        )

        batch = SequenceBatch(
            obs_seqs=self._obs_seqs[idx],
            action_seqs=self._action_seqs[idx],
            reward_seqs=self._rewards_seqs[idx],
            terminal_seqs=self._terminal_seqs[idx],
            next_obs_seqs=self._obs_next_seqs,
        )

        return batch

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])

    def _test_dimensions(self,
                         observation: np.ndarray,
                         action: np.ndarray,
                         reward: np.ndarray,
                         next_observation: np.ndarray,
                         terminal: np.ndarray):

        assert observation.shape \
               == next_observation.shape \
               == (self._observation_dim, self._seq_len)
        assert action.shape == (self._action_dim, self._seq_len)
        assert reward.shape == (1, self._seq_len)
        assert terminal.shape == (1, self._seq_len)

        assert observation.dtype \
               == next_observation.dtype \
               == action.dtype \
               == reward.dtype \
               == np.float64

        assert terminal.dtype == np.uint8

