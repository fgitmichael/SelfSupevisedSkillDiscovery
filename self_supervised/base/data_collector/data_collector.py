import gym
from collections import deque
from typing import List

from rlkit.samplers.data_collector.base import PathCollector
from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.policies.base import Policy
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

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length: int,
            num_steps: int,
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

        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(max_path_length,
                                            num_steps - num_steps_collected)

            path = self._rollouter.do_rollout(
                max_path_length=max_path_length_this_loop,
            )

            path_len = len(path.actions)

            if path_len != max_path_length \
                and not path.terminals[-1] \
                and discard_incomplete_paths:

                break

            num_steps_collected += path_len
            paths.append(path)

        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)

    def get_epoch_paths(self) -> List[TransitionMapping]:
        return list(self._epoch_paths)
