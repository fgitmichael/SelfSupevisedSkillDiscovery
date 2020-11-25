import numpy as np
import os
import torch
from collections import OrderedDict

from self_supervised.base.replay_buffer.replay_buffer_base import SequenceReplayBufferSampleWithoutReplace
from self_supervised.utils.typed_dicts import TransitionMapping


class NormalSequenceReplayBuffer(SequenceReplayBufferSampleWithoutReplace):

    def __init__(self,
                 max_replay_buffer_size,
                 observation_dim,
                 action_dim,
                 seq_len,
                 env_info_sizes=None):
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size
        )

        if env_info_sizes is None:
            env_info_sizes = dict()

        self._observation_dim = observation_dim
        self._action_dim = action_dim

        self._seq_len = seq_len

        self._obs_seqs = np.zeros(
            (self._max_replay_buffer_size,
             self._observation_dim,
             self._seq_len,),
            dtype=np.float32
        )
        self._obs_next_seqs = np.zeros(
            (self._max_replay_buffer_size,
             self._observation_dim,
             self._seq_len),
            dtype=np.float32
        )
        self._action_seqs = np.zeros(
            (self._max_replay_buffer_size,
             self._action_dim,
             self._seq_len),
            dtype=np.float32
        )
        self._rewards_seqs = np.zeros(
            (self._max_replay_buffer_size,
             1,
             self._seq_len),
            dtype=np.float32
        )
        self._terminal_seqs = np.zeros(
            (self._max_replay_buffer_size,
             1,
            self._seq_len),
            dtype=np.uint8
        )

        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros(
                (self._max_replay_buffer_size,
                 size,
                 self._seq_len)
            )
        self._env_info_keys = env_info_sizes.keys()

    def process_save_dict(self, save_obj):
        self._obs_seqs = save_obj['obs']
        self._obs_next_seqs = save_obj['next_obs']
        self._action_seqs = save_obj['actions']
        self._rewards_seqs = save_obj['rewards']
        self._terminal_seqs = save_obj['terminals']
        super().process_save_dict(save_obj)

    def create_save_dict(self) -> dict:
        save_obj = super().create_save_dict()
        save_obj_add = dict(
            obs=self._obs_seqs,
            next_obs=self._obs_next_seqs,
            actions=self._action_seqs,
            rewards=self._rewards_seqs,
            terminals=self._terminal_seqs,
        )
        return dict(
            **save_obj,
            **save_obj_add,
        )


    def add_sample(self,
                   path: TransitionMapping,
                   **kwargs):
        """
        Args:
            path           : TransitionMapping consiting of (dim, S) np.ndarrays
        """
        self._test_dimensions(
            observation=path.obs,
            action=path.action,
            reward=path.reward,
            next_observation=path.next_obs,
            terminal=path.terminal)

        self._obs_seqs[self._top] = path.obs
        self._action_seqs[self._top] = path.action
        self._obs_next_seqs[self._top] = path.next_obs
        self._rewards_seqs[self._top] = path.reward
        self._terminal_seqs[self._top] = path.terminal

        self._advance()

    def terminate_episode(self):
        pass

    def random_batch(self, batch_size: int) -> TransitionMapping:
        idx = np.random.randint(
            low=0,
            high=self._size,
            size=batch_size
        )

        batch = TransitionMapping(
            obs=self._obs_seqs[idx],
            action=self._action_seqs[idx],
            reward=self._rewards_seqs[idx],
            terminal=self._terminal_seqs[idx],
            next_obs=self._obs_next_seqs[idx],
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

        check_type = lambda array_, dtype_: array_.dtype == dtype_
        assert check_type(observation, np.float64) \
               or check_type(observation, np.float32)

        assert check_type(next_observation, np.float64) \
               or check_type(next_observation, np.float32)

        assert check_type(reward, np.float64) \
               or check_type(reward, np.float32)

        assert check_type(action, np.float64) \
               or check_type(action, np.float32)

        assert terminal.dtype == np.bool

    def random_batch_bsd_format(self, batch_size):
        batch_dim = 0
        seq_dim = -1
        data_dim = 1
        return self.random_batch(batch_size).transpose(batch_dim, seq_dim, data_dim)
