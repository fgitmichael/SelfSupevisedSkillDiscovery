from gym import Env
from gym.spaces import Discrete
import numpy as np

from self_supervised.base.replay_buffer.replay_buffer import SequenceReplayBuffer, SequenceBatch, NormalSequenceReplayBuffer
from rlkit.envs.env_utils import get_dim



class SequenceEnvReplayBuffer(NormalSequenceReplayBuffer):

    def __init__(self,
                 max_replay_buffer_size: int,
                 seq_len: int,
                 env: Env,
                 env_info_sizes=None,
                 ):

        self._env = env
        self._obs_space = self._env.observation_space
        self._action_space = self._env.action_space

        if env_info_sizes is None:
            if hasattr(self._env, 'info_sizes'):
                env_info_size = self._env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            seq_len=seq_len,
            observation_dim=get_dim(self._obs_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
        )


    def add_sample(self,
                   observation: np.ndarray,
                   action: np.ndarray,
                   reward: np.ndarray,
                   next_observation: np.ndarray,
                   terminal: np.ndarray,
                   **kwargs):
        self._test_dimensions(
            observation = observation,
            action = action,
            reward = reward,
            next_observation = next_observation,
            terminal = terminal)

        if isinstance(self._action_space, Discrete):
            # One Hot over sequences
            new_action = np.zeros(
                (action.shape[0], self._action_dim, self._seq_len)
            )
            new_action[action] = 1
        else:
            new_action = action

        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )
