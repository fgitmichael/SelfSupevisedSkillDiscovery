import torch
import gym
import numpy as np
from collections import deque
from typing import List, Union

from rlkit.torch.sac.diayn.policies import MakeDeterministic

import self_supervised.utils.typed_dicts as td

from diayn_seq_code_revised.base.data_collector_base import PathCollectorRevisedBase
from diayn_seq_code_revised.policies.skill_policy import \
    SkillTanhGaussianPolicyRevised, MakeDeterministicRevised
from diayn_seq_code_revised.data_collector.rollouter_revised import RollouterRevised
from diayn_seq_code_revised.base.skill_selector_base import SkillSelectorBase
from diayn_seq_code_revised.data_collector.skill_selector import SkillSelectorDiscrete
from diayn_seq_code_revised.base.rollouter_base import RollouterBase

import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.rollout_functions import rollout


class SeqCollectorRevised(PathCollectorRevisedBase):

    def __init__(self,
                 env: gym.Env,
                 policy: Union[
                     SkillTanhGaussianPolicyRevised,
                     MakeDeterministicRevised
                 ],
                 skill_selector: SkillSelectorBase,
                 max_seqs: int,
                 reset_env_after_collection=False,
                 ):
        self.policy = policy
        self._rollouter = self.create_rollouter(
            env=env,
            policy=self.policy,
            reset_env_after_collection=reset_env_after_collection,
        )
        self.skill_selector = skill_selector

        self._epoch_paths = deque(maxlen=max_seqs)
        self._skill = None
        self._num_steps_total = 0
        self._num_paths_total = 0

    @property
    def maxlen(self):
        return self._epoch_paths.maxlen

    def create_rollouter(
            self,
            env,
            policy,
            reset_env_after_collection=False,
    ) -> RollouterBase:
        return RollouterRevised(
            env=env,
            policy=policy,
            reset_env_after_collection=reset_env_after_collection,
        )

    @property
    def skill(self):
        return self.policy.skill

    @skill.setter
    def skill(self, skill: torch.Tensor):
        self.policy.skill = skill

    def skill_reset(self):
        random_skill = self.skill_selector.get_random_skill()
        self.skill = random_skill

    def reset(self):
        self._epoch_paths = deque(maxlen=self._epoch_paths.maxlen)
        self._rollouter.reset()

    def _collect_new_paths(self,
                           num_seqs: int,
                           seq_len: int,
                           obs_dim_to_select: list) -> List[td.TransitionMapping]:
        paths = []
        num_steps_collected = 0

        for _ in range(num_seqs):
            path = self._rollouter.do_rollout(
                seq_len=seq_len
            )
            self._check_paths(path=path, seq_len=seq_len)

            if obs_dim_to_select is not None:
                path = self.select_obs_dims(
                    path,
                    obs_dims_to_select=obs_dim_to_select
                )

            num_steps_collected += seq_len
            paths.append(path)

        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected

        return paths

    def _check_path(self, path, seq_len):
        assert path.obs.shape == (seq_len, self.obs_dim)

    def collect_new_paths(
            self,
            seq_len,
            num_seqs,
            discard_incomplete_paths=None,
            obs_dim_to_select=None,
    ):
        paths = self._collect_new_paths(
            seq_len=seq_len,
            num_seqs=num_seqs,
            obs_dim_to_select=obs_dim_to_select
        )
        self._check_path(paths[0], seq_len)

        prepared_paths = self.prepare_paths_before_save(paths, seq_len)
        self._epoch_paths.extend(prepared_paths)

    def prepare_paths_before_save(self, paths, seq_len) \
            -> List[td.TransitionModeMapping]:
        # Extend to TransitionModeMapping
        seq_dim = 0
        skill_seq = np.stack(
            [ptu.get_numpy(self.skill)] * seq_len,
            axis=seq_dim
        )
        assert skill_seq.shape == (seq_len, self.skill_selector.skill_dim)

        paths_with_skills = []
        for (idx, path) in enumerate(paths):
            with_skill = td.TransitionModeMapping(
                **path,
                mode=skill_seq
            )
            paths_with_skills.append(with_skill)

        return paths_with_skills

    def select_obs_dims(
            self,
            paths: td.TransitionMapping,
            obs_dims_to_select: Union[list, tuple],
    ) -> td.TransitionMapping:
        data_dim = -1
        assert paths.obs.shape[data_dim] == self.obs_dim
        assert isinstance(obs_dims_to_select, list) \
               or isinstance(obs_dims_to_select, tuple)

        paths.obs = paths.obs[..., obs_dims_to_select]
        paths.next_obs = paths.next_obs[..., obs_dims_to_select]

        return paths

    @property
    def obs_dim(self):
        return self._rollouter.env.observation_space.shape[-1]

    @property
    def action_dim(self):
        return self._rollouter.env.action_space.shape[-1]

    def _check_paths(self,
                     seq_len,
                     path: td.TransitionMapping):
        obs_dim = self.obs_dim
        action_dim = self._rollouter.env.action_space.shape[-1]
        skill_dim = self.skill_selector.skill_dim

        batch_dim = 0
        seq_dim = -2
        shape_dim = -1

        assert len(path.obs.shape) \
               == len(path.next_obs.shape) \
               == len(path.action.shape) \
               == len(path.terminal.shape) \
               == len(path.reward.shape) \

        assert path.action.shape[shape_dim] == action_dim
        assert path.obs.shape[shape_dim] \
               == path.next_obs.shape[shape_dim] \
               == obs_dim
        assert path.action.shape[seq_dim] \
               == path.obs.shape[seq_dim] \
               == path.reward.shape[seq_dim] \
               == path.terminal.shape[seq_dim] \
               == path.next_obs.shape[seq_dim] \
               == seq_len
        if len(path.obs.shape) > 2:
            assert path.action.shape[batch_dim] \
                   == path.obs.shape[batch_dim] \
                   == path.reward.shape[batch_dim] \
                   == path.terminal.shape[batch_dim] \
                   == path.next_obs.shape[batch_dim] \
                   == path.mode.shape[batch_dim]

    def get_epoch_paths(self, reset=True, transpose=True) \
            -> Union[
                List[td.TransitionModeMapping],
                List[td.TransitonModeMappingDiscreteSkills]]:
        """
        Return:
            list of TransistionMapping consisting of (S, data_dim) np.ndarrays
        """
        assert len(self._epoch_paths) > 0
        epoch_paths = list(self._epoch_paths)

        if transpose:
            for idx, path in enumerate(epoch_paths):
                assert len(path.obs.shape) == 2
                epoch_paths[idx] = path.transpose(1, 0)

        if reset:
            self.reset()

        return epoch_paths

    def set_skill(self, skill):
        self.skill = skill
