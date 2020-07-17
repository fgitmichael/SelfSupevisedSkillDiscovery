import numpy as np
from typing import List
from prodict import Prodict


from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer
import self_supervised.utils.typed_dicts as td

class TransitonModeMappingDiscreteSkills(td.TransitionModeMapping):
    skill_id: np.ndarray

    def __init__(self,
                 obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 terminal: np.ndarray,
                 next_obs: np.ndarray,
                 mode: np.ndarray,
                 skill_id: np.ndarray,
                 agent_infos=None,
                 env_infos=None,
                 ):

        Prodict.__init__(
            self,
            obs=obs,
            action=action,
            reward=reward,
            terminal=terminal,
            next_obs=next_obs,
            mode=mode,
            skill_id=skill_id,
            agent_infos=agent_infos,
            env_infos=env_infos
        )


class SelfSupervisedEnvSequenceReplayBufferDiscreteSkills(
    SelfSupervisedEnvSequenceReplayBuffer):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._skill_id_per_seq = np.zeros(
            (self._max_replay_buffer_size,
             1,
             self._seq_len),
            dtype=np.uint8
        )

    def add_sample(self,
                   path: TransitonModeMappingDiscreteSkills,
                   **kwargs):
        self._skill_id_per_seq[self._top] = path.pop('skill_id')

        super().add_sample(
            path=td.TransitionModeMapping(**path),
            **kwargs
        )

    def add_self_sup_paths(self,
                           paths: List[TransitonModeMappingDiscreteSkills]):
        for path in paths:
            self.add_sample(path)

    def random_batch(self,
                     batch_size: int) -> TransitonModeMappingDiscreteSkills:
        """
        Args:
            batch_size                 : N
        Return:
            TransitionModeMapping      : consisting of (N, data_dim, S) tensors
        """
        idx = np.random.randint(0, self._size, batch_size)

        batch = TransitonModeMappingDiscreteSkills(
            obs=self._obs_seqs[idx],
            action=self._action_seqs[idx],
            reward=self._rewards_seqs[idx],
            next_obs=self._obs_next_seqs[idx],
            terminal=self._terminal_seqs[idx],
            mode=self._mode_per_seqs[idx],
            skill_id=self._skill_id_per_seq[idx]
        )

        return batch


