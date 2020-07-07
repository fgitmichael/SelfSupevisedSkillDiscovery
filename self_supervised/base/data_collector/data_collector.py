import gym
import numpy as np
from collections import deque
from typing import List

from rlkit.samplers.data_collector.base import PathCollector
from rlkit.torch.sac.diayn.policies import SkillTanhGaussianPolicy

from self_supervised.base.data_collector.rollout import Rollouter
from self_supervised.utils.typed_dicts import TransitionMapping


class PathCollectorSelfSupervised(PathCollector):

    def __init__(self,
                 env: gym.Env,
                 policy: SkillTanhGaussianPolicy,
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

        self.obs_dim = env.observation_space[0]
        self.action_dim = env.action_space[0]

        self._num_steps_total = 0
        self._num_paths_total = 0
        self.seq_len = 0


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
            for i, k in path:
                assert len(path[i].shape) == 2
            assert path.action.shape[-1] == self._rollouter._env.action_space[0]
            assert path.obs.shape[-1] \
                   == path.next_obs.shape[-1] \
                   == self._rollouter._env.observation_space[0]
            assert path.action.shape[-2] \
                   == path.obs.shape[-2] \
                   == path.reward.shape[-2] \
                   == path.terminal.shape[-2] \
                   == path.next_obs.shape[-2] \
                   == seq_len

            num_steps_collected += seq_len
            paths.append(path)

        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)

    def get_epoch_paths(self) -> List[TransitionMapping]:
        """
        Return:
            list of TransistionMapping consisting of (N, dim, S) np.ndarrays
        """
        return list(self._epoch_paths)
