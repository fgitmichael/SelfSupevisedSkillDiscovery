from gym import Env
from gym.spaces import Discrete
import numpy as np

from self_supervised.base.replay_buffer.replay_buffer \
    import SequenceReplayBuffer, NormalSequenceReplayBuffer
from self_supervised.utils.typed_dicts import TransitionMapping
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
                   sample: TransitionMapping,
                   **kwargs):

        if isinstance(self._action_space, Discrete):
            # One Hot over sequences
            new_action = np.zeros(
                (self._action_dim, self._seq_len)
            )
            new_action[sample.action_seqs] = 1
        else:
            new_action = sample.action_seqs
        sample.action_seqs = new_action

        super().add_sample(
            sample=sample
            **kwargs
        )
