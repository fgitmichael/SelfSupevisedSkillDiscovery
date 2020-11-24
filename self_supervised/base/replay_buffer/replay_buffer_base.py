import abc
import os
import torch
from typing import List
import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer

import self_supervised.utils.typed_dicts as td

from latent_with_splitseqs.base.my_object_base import MyObjectBase

# Adding skills
class SequenceReplayBuffer(ReplayBuffer, MyObjectBase, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def add_sample(self,
                   observation,
                   action,
                   reward,
                   next_observation,
                   terminal,
                   **kwargs):
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def add_path(self, path: td.TransitionMapping):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info,
                skills
        ) in enumerate(zip(
            path.obs,
            path.action,
            path.reward,
            path.next_obs,
            path.terminal,
            path.agent_infos,
            path.env_infos,
            path.mode
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
                skill=skills
            )
        self.terminate_episode()

    def add_paths(self, paths: List[td.TransitionModeMapping]):
        assert paths is not None
        for path in paths:
            self.add_path(path)

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return


class SequenceReplayBufferSampleWithoutReplace(SequenceReplayBuffer):

    def __init__(self,
                 max_replay_buffer_size):
        self._max_replay_buffer_size = max_replay_buffer_size
        self._size = 0
        self._top = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size

        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def __len__(self):
        return self._size

    def _get_sample_idx(self, batch_size):
        if self._size < batch_size:
            idx_present = np.arange(self._size)
            idx_rest = np.random.randint(
                low=0,
                high=self._size,
                size=batch_size-self._size
            )
            idx = np.concatenate(
                [idx_present, idx_rest]
            )

        else:
            idx = np.random.choice(
                np.arange(self._size),
                size=batch_size,
                replace=False
            )

        return idx
