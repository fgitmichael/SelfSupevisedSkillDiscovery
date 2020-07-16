import gym
import numpy as np
from collections import deque
from typing import List, Union

from rlkit.samplers.data_collector.base import PathCollector

from self_supervised.base.data_collector.rollout import Rollouter
from self_supervised.policy.skill_policy import MakeDeterministic, SkillTanhGaussianPolicy
import self_supervised.utils.typed_dicts as td


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

        self._rollouter = Rollouter(
            env=env,
            policy=policy
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
            discard_incomplete_paths: bool,
    ):
        """
        Args:
            num_steps                  : int i.e. num_eval_steps_per_epoch
                                         (typically higher
                                         than max_path_length, typically a multiply of
                                         max_path_length)
            max_path_length            : maximal path length
            discard_incomplete_paths   : path

        Return:
            paths                      : deque
        """
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
        return list(self._epoch_paths)

    def end_epoch(self, epoch):
        super().end_epoch(epoch)

        self._rollouter.reset()
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

