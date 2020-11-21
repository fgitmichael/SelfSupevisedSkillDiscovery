import torch
import gym
import numpy as np
from collections import deque
from typing import List, Union

from latent_with_splitseqs.base.split_seq_collector_horizon_base \
    import HorizonSplitSeqCollectorBase
from latent_with_splitseqs.data_collector.rollout_without_reset \
    import CollectSeqOverHorizonWrapper

from diayn_seq_code_revised.policies.skill_policy import \
    SkillTanhGaussianPolicyRevised, MakeDeterministicRevised
from diayn_seq_code_revised.base.skill_selector_base import SkillSelectorBase
from diayn_seq_code_revised.base.rollouter_base import RollouterBase
from diayn_seq_code_revised.data_collector.rollouter_revised import RollouterRevised

import self_supervised.utils.typed_dicts as td

import rlkit.torch.pytorch_util as ptu


class SeqCollectorHorizon(HorizonSplitSeqCollectorBase):

    def __init__(
            self,
            env: gym.Env,
            policy: Union[
                SkillTanhGaussianPolicyRevised,
                MakeDeterministicRevised,
            ],
            skill_selector: SkillSelectorBase,
            max_seqs: int = 5000,
    ):
        self.max_seqs = max_seqs
        self._epoch_split_seqs = None
        super(SeqCollectorHorizon, self).__init__(
            env=env,
            policy=policy,
        )
        self.policy = policy
        self.skill_selector = skill_selector
        self._skill = None
        self._obs = None

        self._num_steps_total = 0
        self._num_split_seqs_total = 0
        self._num_split_seqs_current_rollout = 0

    def create_rollouter(
            self,
            env,
            policy,
            **kwargs
    ) -> RollouterBase:
        rollout_wrapper = CollectSeqOverHorizonWrapper()
        return RollouterRevised(
            env=env,
            policy=policy,
            rollout_wrapper=rollout_wrapper
        )

    def reset(self):
        self._epoch_split_seqs = deque(maxlen=self.max_seqs)
        self._num_split_seqs_current_rollout = 0
        self._rollouter.reset()

    def skill_reset(self):
        random_skill = self.skill_selector.get_random_skill()
        self.skill = random_skill

    def collect_split_seq(
            self,
            seq_len,
            horizon_len,
            discard_incomplete_seq,
    ) -> bool:
        seq = self._rollouter.do_rollout(
            seq_len=seq_len
        )
        seq_with_skill = self.prepare_seq_before_save(
            seq=seq,
            sampling_seq_len=seq_len,
            discard_incomplete_seq=discard_incomplete_seq
        )

        seq_dim = 0
        sampled_seq_len = seq.obs.shape[seq_dim]
        self._num_steps_total += sampled_seq_len
        self._num_split_seqs_total += 1
        self._num_split_seqs_current_rollout += 1

        # Reset rollouter
        num_seqs_horizon = horizon_len // seq_len
        horizon_completed = False
        if self._num_split_seqs_current_rollout == num_seqs_horizon:
            self._rollouter.reset()
            horizon_completed = True

        self._epoch_split_seqs.append(seq_with_skill)

        return horizon_completed

    @property
    def skill(self):
        return self.policy.skill

    @skill.setter
    def skill(self, skill: torch.Tensor):
        self.policy.skill = skill

    @property
    def obs_dim(self):
        return self._rollouter.env.observation_space.shape[-1]

    @property
    def action_dim(self):
        return self._rollouter.env.action_space.shape[-1]

    def prepare_seq_before_save(
            self,
            seq,
            sampling_seq_len,
            discard_incomplete_seq
    ) -> td.TransitionModeMapping:
        # Handle possible incomplete path
        seq_dim = 0
        sampled_seq_len = seq.obs.shape[seq_dim]
        seq_complete = sampled_seq_len == sampling_seq_len
        assert sampling_seq_len >= sampled_seq_len
        if discard_incomplete_seq and not seq_complete:
            prepared_seq = None

        elif not discard_incomplete_seq and not seq_complete:
            # Copy the last (sampling_seq_len - seq_len) elements
            # to complete the sequence
            assert len(self._epoch_split_seqs) > 0
            copy_len = sampling_seq_len - sampled_seq_len
            last_seq = self._epoch_split_seqs[-1]
            completed_seq = {}
            seq_dim = 0
            for key, el in last_seq.items():
                if isinstance(el, np.ndarray):
                    completed_seq[key] = np.concatenate(
                        (el[-copy_len:], seq[key]),
                        axis=seq_dim,
                    )
                    assert completed_seq[key].shape[seq_dim] == sampling_seq_len

            prepared_seq = self.extend_transitions_with_skill(
                seq=seq,
                sampling_seq_len=sampling_seq_len,
            )

        else:
            # Seq is complete
            prepared_seq = self.extend_transitions_with_skill(
                seq=seq,
                sampling_seq_len=sampling_seq_len
            )

        return prepared_seq

    def extend_transitions_with_skill(
            self,
            seq: td.TransitionMapping,
            sampling_seq_len,
    ) -> td.TransitionModeMapping:
        seq_len = sampling_seq_len
        # Extend to TransitionModeMapping
        seq_dim = 0
        skill_seq = np.stack(
            [ptu.get_numpy(self.skill)] * seq_len,
            axis=seq_dim
        )
        assert skill_seq.shape == (seq_len, self.skill_selector.skill_dim)
        seq_with_skill = td.TransitionModeMapping(
            **seq,
            mode=skill_seq,
        )

        return seq_with_skill

    def get_epoch_paths(
            self,
            reset=True,
            transpose=True) \
            -> List[td.TransitionModeMapping]:
        """
        Return:
            list of TransistionMapping consisting of (S, data_dim) np.ndarrays
        """
        assert len(self._epoch_split_seqs) > 0
        epoch_seqs = list(self._epoch_split_seqs)

        if transpose:
            for idx, seq in enumerate(epoch_seqs):
                assert len(seq.obs.shape) == 2
                epoch_seqs[idx] = seq.transpose(1, 0)

        if reset:
            self.reset()

        return epoch_seqs

    def set_skill(self, skill):
        self.skill = skill
