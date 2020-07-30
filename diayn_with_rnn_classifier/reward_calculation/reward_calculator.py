import torch
import math
import torch.nn as nn
from typing import Union

from diayn_original_tb.policies.self_sup_policy_wrapper import \
    MakeDeterministicExtension, MakeDeterministicMyPolicyWrapper

import rlkit.torch.pytorch_util as ptu
import self_supervised.utils.my_pytorch_util as my_ptu


class RewardPolicyDiff():

    def __init__(self,
                 eval_policy: Union[
                     MakeDeterministicExtension,
                     MakeDeterministicMyPolicyWrapper]
                 ):
        self.eval_policy = eval_policy

    @torch.no_grad()
    def calc_rewards(self,
                     obs_seq: torch.Tensor,
                     action_seq: torch.Tensor,
                     skill_gt_id: torch.Tensor,
                     pred_log_softmax: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_seq          : (N, S, obs_dim) tensor
            action_seq       : (N, S, action_dim) tensor
            skill_gt_id      : (N, 1) oh tensor
            pred_log_softmax : (N, skill_dim) "oh" tensor
        Return:
            rewards          : (N, S, 1) tensor
        """
        base_rewards = self._calc_base_rewards(
            obs_seq=obs_seq,
            skill_gt_id=skill_gt_id,
            pred_log_softmax=pred_log_softmax
        )

        specific_rewards = self._calc_transition_specific_rewards(
            obs_seq=obs_seq,
            skill_gt_id=skill_gt_id,
            action_seq=action_seq,
            pred_log_softmax=pred_log_softmax
        )

        assert base_rewards.shape \
            == specific_rewards.shape \
            == torch.Size((obs_seq.size(0), obs_seq.size(1), 1))

        return base_rewards - specific_rewards

    def _calc_transition_specific_rewards(
            self,
            obs_seq: torch.Tensor,
            action_seq: torch.Tensor,
            skill_gt_id: torch.Tensor,
            pred_log_softmax: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_seq          : (N, S, obs_dim) tensor
            action_seq       : (N, S, action_dim) tensor
            skill_gt_id      : (N, 1) oh tensor
            pred_log_softmax : (N, skill_dim) "oh" tensor
        Return:
            specific_rewards : (N, S, 1) tensor
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1

        batch_size = obs_seq.size(batch_dim)
        seq_len = obs_seq.size(seq_dim)
        skill_dim = pred_log_softmax.size(data_dim)

        pred_skill_id = self._get_pred_skill(pred_log_softmax)
        assert pred_skill_id.shape == torch.Size((batch_size, 1))

        skill_gt_id_seq = self._repeat_tensor(
            tensor=skill_gt_id,
            reps=seq_len,
            dim=seq_dim
        )
        pred_skill_id_seq = self._repeat_tensor(
            tensor=pred_skill_id,
            reps=seq_len,
            dim=seq_dim
        )

        obs_stacked = obs_seq.view(-1, obs_seq.size(data_dim))
        skill_gt_id_stacked = skill_gt_id_seq.view(-1, 1)
        pred_skill_id_stacked = pred_skill_id_seq.view(-1, 1)
        assert obs_stacked.size(batch_dim) \
               == skill_gt_id_stacked.size(batch_dim) \
               == pred_skill_id_stacked.size(batch_dim) \
               == batch_size * seq_len

        (_,
         _,
         _,
         log_prob_gt,
         _,
         _,
         _,
         _) = self.eval_policy.forward(
            obs=obs_stacked,
            skill_vec=self._get_skill_from_id(skill_gt_id_stacked, skill_dim),
            reparameterize=False,
            return_log_prob=True
        )

        (_,
         _,
         _,
         log_prob_pred,
         _,
         _,
         _,
         _) = self.eval_policy.forward(
            obs=obs_stacked,
            skill_vec=self._get_skill_from_id(pred_skill_id_stacked, skill_dim),
            reparameterize=False,
            return_log_prob=True
        )

        assert log_prob_gt.shape \
               == log_prob_pred.shape \
               == torch.Size((batch_size, action_seq.size(data_dim)))

        log_prob_gt_unstacked = log_prob_gt.view(batch_size, seq_len, 1)
        log_prob_pred_unstacked = log_prob_pred.view(batch_size, seq_len, 1)
        assert log_prob_gt[:seq_len, :] == log_prob_gt_unstacked[0, :, :]
        assert log_prob_pred[:seq_len, :] == log_prob_pred_unstacked[0, :, :]

        return (log_prob_gt_unstacked - log_prob_pred_unstacked)**2

    def _get_skill_from_id(self, skill_id, skill_dim):
        """
        Args:
            skill_id            : (N, 1) tensor
        """
        # skill_dim is needed cause it is same as number of classes here
        num_classes = skill_dim
        return my_ptu.eye(num_classes)[skill_id.squeeze()]

    def _get_pred_skill(self, pred_log_softmax: torch.Tensor) -> torch.Tensor:
        pred_skill_id = torch.argmax(pred_log_softmax, dim=-1, keepdim=True)
        return pred_skill_id

    def _repeat_tensor(self, tensor: torch.Tensor, reps, dim):
        return torch.stack([tensor] * reps, dim=dim)

    def _calc_base_rewards(self,
                           obs_seq: torch.Tensor,
                           skill_gt_id: torch.Tensor,
                           pred_log_softmax: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_seq          : (N, S, obs_dim) tensor
            skill_gt_id      : (N, 1) oh tensor
            pred_log_softmax : (N, skill_dim) "oh" tensor
        Return:
            base_rewards     : (N, S, 1) tensor
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1

        batch_size = obs_seq.size(batch_dim)
        seq_len = obs_seq.size(seq_dim)
        skill_dim = pred_log_softmax.size(data_dim)

        assert skill_gt_id.shape == torch.Size((batch_size, 1))

        pred_skill_id = self._get_pred_skill(pred_log_softmax)
        assert pred_skill_id.shape == torch.Size((batch_size, 1))

        base_rewards = pred_log_softmax[torch.arange(batch_size), skill_gt_id.squeeze()] \
                       - math.log(1/skill_dim)
        assert base_rewards.shape == torch.Size((batch_size,))
        base_rewards = base_rewards.unsqueeze(dim=1)
        base_rewards = self._repeat_tensor(
            tensor=base_rewards,
            reps=seq_len,
            dim=seq_dim
        )
        assert base_rewards.shape == torch.Size((batch_size, seq_len, 1))

        return base_rewards
