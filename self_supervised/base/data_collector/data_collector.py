import gym
import numpy as np
from collections import deque
from typing import List, Union

from rlkit.samplers.data_collector.base import PathCollector

from self_supervised.base.data_collector.rollout import Rollouter
from self_supervised.policy.skill_policy import MakeDeterministic, SkillTanhGaussianPolicy
import self_supervised.utils.typed_dicts as td

from self_sup_combined.discrete_skills.replay_buffer_discrete_skills import TransitonModeMappingDiscreteSkills


class PathCollectorSelfSupervised(PathCollector):

    def __init__(self,
                 env: gym.Env,
                 policy: Union[SkillTanhGaussianPolicy, MakeDeterministic],
                 max_num_epoch_paths_saved: int = None,
                 render: bool = False,
                 render_kwargs: bool = None
                 ):
        if render_kwargs is None:
            render_kwargs = {}

        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self.policy = policy
        self._rollouter = Rollouter(
            env=env,
            policy=self.policy
        )

        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self._num_steps_total = 0
        self._num_paths_total = 0
        self.seq_len = None

    def collect_new_paths(
            self,
            seq_len: int,
            num_seqs: int,
            discard_incomplete_paths: bool=False,
    ):
        """
        Args:
            seq_len                    : sequence length
            num_seqs                   : number of sequence to sample
            discard_incomplete_paths   : path

        Return:
            paths                      : deque
        """
        paths = self._collect_new_paths(
            seq_len=seq_len,
            num_seqs=num_seqs,
            discard_incomplete_paths=discard_incomplete_paths
        )

        self._epoch_paths.extend(paths)

    def _collect_new_paths(self,
                          seq_len: int,
                          num_seqs: int,
                          discard_incomplete_paths: bool
                          ) -> List[td.TransitionModeMapping]:
        paths = []
        num_steps_collected = 0
        self.seq_len = seq_len

        for _ in range(num_seqs):

            path = self._rollouter.do_rollout(
                max_path_length=seq_len,
            )

            assert len(path.obs.shape) \
                == len(path.next_obs.shape) \
                == len(path.action.shape) \
                == len(path.terminal.shape) \
                == len(path.reward.shape) \
                == len(path.mode.shape)

            batch_dim = 0
            shape_dim = -2
            seq_dim = -1
            assert path.action.shape[shape_dim] \
                == self._rollouter._env.action_space.shape[0]
            assert path.obs.shape[shape_dim] \
                == path.next_obs.shape[shape_dim] \
                == self._rollouter._env.observation_space.shape[0]
            assert path.mode.shape[shape_dim] == self._rollouter._real_policy.skill_dim
            assert path.action.shape[seq_dim] \
                == path.obs.shape[seq_dim] \
                == path.reward.shape[seq_dim] \
                == path.terminal.shape[seq_dim] \
                == path.next_obs.shape[seq_dim] \
                == path.mode.shape[seq_dim] \
                == seq_len
            if len(path.obs.shape) > 2:
                assert path.action.shape[batch_dim] \
                    == path.obs.shape[batch_dim] \
                    == path.reward.shape[batch_dim] \
                    == path.terminal.shape[batch_dim] \
                    == path.next_obs.shape[batch_dim] \
                    == path.mode.shape[batch_dim]

            num_steps_collected += seq_len
            paths.append(path)

        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)

    def get_epoch_paths(self) -> List[td.TransitionModeMapping]:
        """
        Return:
            list of TransistionMapping consisting of (S, dim) np.ndarrays
        """
        epoch_paths = list(self._epoch_paths)
        self.reset()

        return epoch_paths

    def reset(self):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._rollouter.reset()

    def end_epoch(self, epoch, reset=False):
        super().end_epoch(epoch)

        # Reset is already done, when popping epoch paths
        # Note: without popping epoch paths now reset is done by default
        if reset:
            self.reset()


class  PathCollectorSelfSupervisedDiscreteSkills(PathCollectorSelfSupervised):

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.skill_id = None
