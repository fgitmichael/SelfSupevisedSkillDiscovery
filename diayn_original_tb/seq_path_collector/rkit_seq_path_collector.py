import torch
import gym
import numpy as np
from collections import deque
from typing import List, Union

from rlkit.samplers.data_collector.base import PathCollector
from rlkit.torch.sac.diayn.policies import MakeDeterministic

import self_supervised.utils.typed_dicts as td

from diayn_original_tb.seq_path_collector.rlkit_rollouter import Rollouter
from diayn_original_tb.policies.diayn_policy_extension import \
    SkillTanhGaussianPolicyExtension, MakeDeterministicExtension

from diayn_no_oh.policies.diayn_policy_no_oh import SkillTanhGaussianPolicyNoOHTwoDim


class SeqCollector(PathCollector):

    def __init__(self,
                 env: gym.Env,
                 policy: Union[
                         SkillTanhGaussianPolicyExtension,
                         SkillTanhGaussianPolicyNoOHTwoDim,
                         MakeDeterministicExtension],
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

        self._skill = None
        self._rollouter = Rollouter(
            env=env,
            policy=policy
        )

        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.observation_space.shape[0]

        self._num_steps_total = 0
        self._num_paths_total = 0
        self.seq_len = None

    def set_skill(self, skill: int):
        """
        Args:
            skill       : integer (0 < skill < skill_dim)
        """
        self._skill = skill

    def _collect_new_paths(self,
                           seq_len: int,
                           num_seqs: int,
                           discard_incomplete_paths: bool,
                           ) -> List[td.TransitionModeMapping]:
        paths = []
        num_steps_collected = 0
        self.seq_len = seq_len

        for _ in range(num_seqs):

            path = self._rollouter.do_rollout(
                skill=self._skill,
                max_path_length=seq_len,
            )

            self._check_paths(path)

            num_steps_collected += seq_len
            paths.append(path)

        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected

        return paths

    def collect_new_paths(self,
                          seq_len: int,
                          num_seqs: int,
                          discard_incomplete_paths: bool
                          ):
        paths = self._collect_new_paths(
            seq_len=seq_len,
            num_seqs=num_seqs,
            discard_incomplete_paths=discard_incomplete_paths
        )

        # Extend TransitonModeMapping to TransitionModeMappingDiscreteSkills
        seq_dim = 1
        skill_id_seq = np.stack(
            [np.array([self._rollouter._policy.skill])] * seq_len,
            axis=seq_dim
        )
        assert skill_id_seq.shape[seq_dim] == paths[0].obs.shape[seq_dim]
        for (idx, path) in enumerate(paths):
            with_skill_id = td.TransitonModeMappingDiscreteSkills(
                **path,
                skill_id=skill_id_seq
            )
            paths[idx] = with_skill_id

        self._epoch_paths.extend(paths)

    def _check_paths(self, path: td.TransitionModeMapping):
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
        assert path.mode.shape[shape_dim] == self._rollouter._policy.skill_dim
        assert path.action.shape[seq_dim] \
               == path.obs.shape[seq_dim] \
               == path.reward.shape[seq_dim] \
               == path.terminal.shape[seq_dim] \
               == path.next_obs.shape[seq_dim] \
               == path.mode.shape[seq_dim] \
               == self.seq_len
        if len(path.obs.shape) > 2:
            assert path.action.shape[batch_dim] \
                   == path.obs.shape[batch_dim] \
                   == path.reward.shape[batch_dim] \
                   == path.terminal.shape[batch_dim] \
                   == path.next_obs.shape[batch_dim] \
                   == path.mode.shape[batch_dim]

    def get_epoch_paths(self) -> List[td.TransitonModeMappingDiscreteSkills]:
        """
        Return:
            list of TransistionMapping consisting of (data_dim, S) np.ndarrays
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
