import torch
import abc
import gym
import numpy as np
from collections import deque
from typing import List, Union, Tuple

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


class SeqCollectorHorizonBase(HorizonSplitSeqCollectorBase):

    def __init__(
            self,
            env: gym.Env,
            policy: Union[
                SkillTanhGaussianPolicyRevised,
                MakeDeterministicRevised,
            ],
            skill_selector: SkillSelectorBase,
            max_seqs: int = 5000,
            terminal_handling: bool = False,
    ):
        self.max_seqs = max_seqs
        self._epoch_split_seqs = None
        super().__init__(
            env=env,
            policy=policy,
        )
        self.policy = policy
        self.skill_selector = skill_selector
        self._skill = None
        self._obs = None

        self._terminal_handling_bool = terminal_handling

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

    @abc.abstractmethod
    def _save_split_seq(self, split_seq, horizon_completed: bool):
        raise NotImplementedError

    def collect_split_seq(
            self,
            seq_len,
            horizon_len,
            discard_incomplete_seq,
    ) -> bool:
        # Rollout
        seq = self._rollouter.do_rollout(
            seq_len=seq_len
        )

        # Handle terminals
        if self._terminal_handling_bool:
            seq_terminals_handled, seq_terminated = self._handle_terminals(seq)

        else:
            seq_terminals_handled = seq
            seq_terminated = False

        # Add skills
        seq_with_skills = self._extend_transitions_with_skill(
            seq=seq_terminals_handled,
        )

        # Update counters
        seq_dim = 0
        sampled_seq_len = seq_with_skills.obs.shape[seq_dim]
        self._num_steps_total += sampled_seq_len
        self._num_split_seqs_total += 1
        self._num_split_seqs_current_rollout += 1

        # Reset rollouter
        num_seqs_horizon = horizon_len // seq_len
        horizon_completed = False
        if self._num_split_seqs_current_rollout == num_seqs_horizon or seq_terminated:
            self._num_split_seqs_current_rollout = 0
            self._rollouter.reset()
            horizon_completed = True

        # Decide if incomplete sequence (fragments) are saved
        # (Sequence can be incomplete if terminals occured)
        save_seq = sampled_seq_len == seq_len or seq_terminated
        if save_seq or not discard_incomplete_seq:
            self._save_split_seq(
                split_seq=seq_with_skills,
                horizon_completed=horizon_completed,
            )

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

    def _handle_terminals(self, seq) -> Tuple[td.TransitionMapping, bool]:
        seq_dim = 0

        assert isinstance(seq, td.TransitionMapping)
        terminal = np.squeeze(seq.terminal)
        assert isinstance(terminal, np.ndarray)
        assert len(terminal.shape) == 1
        assert terminal.dtype == np.bool

        seq_terminated = not np.all(terminal == False, axis=seq_dim)

        # Always use at least two sequence chunk,
        # otherwise replay buffer will never be filled
        remove_terminals = self._num_split_seqs_current_rollout > 0 and seq_terminated

        if remove_terminals:
            terminal_idx = np.argwhere(terminal == True)
            first_terminal_idx = int(terminal_idx[0])
            seq_until_terminal = {}
            for key, el in seq.items():
                if isinstance(el, np.ndarray):
                    seq_until_terminal[key] = el[:(first_terminal_idx + 1)]
                else:
                    seq_until_terminal[key] = el
            seq = td.TransitionMapping(**seq_until_terminal)

        return seq, seq_terminated

    def _extend_transitions_with_skill(
            self,
            seq: td.TransitionMapping,
    ) -> td.TransitionModeMapping:

        # Get sampled sequence length
        seq_dim = 0
        seq_lens = [el.shape[seq_dim] for el in seq.values() if isinstance(el, np.ndarray)]
        assert all([path_len == seq_lens[0] for path_len in seq_lens])
        seq_len = seq_lens[0]

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
            transpose=True,
    ) -> List[td.TransitionModeMapping]:
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
