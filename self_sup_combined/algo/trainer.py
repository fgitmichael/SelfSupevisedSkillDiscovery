import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Dict

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import FlattenMlp

from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.policy.skill_policy import SkillTanhGaussianPolicy
from self_supervised.base.trainer.trainer_base import MyTrainerBaseClass
import self_supervised.utils.conversion as self_sup_conversion
import self_supervised.utils.typed_dicts as td

from self_sup_combined.network.mode_encoder import ModeEncoderSelfSupComb
from self_sup_combined.loss.mode_likelihood_based_reward import \
    ReconstructionLikelyhoodBasedRewards


class SelfSupCombSACTrainer(MyTrainerBaseClass):

    def __init__(self,
                 env: NormalizedBoxEnvWrapper,
                 policy: SkillTanhGaussianPolicy,
                 qf1: FlattenMlp,
                 qf2: FlattenMlp,
                 target_qf1: FlattenMlp,
                 target_qf2: FlattenMlp,

                 intrinsic_reward_calculator: ReconstructionLikelyhoodBasedRewards,
                 discount=0.99,
                 reward_scale=1.0,

                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 optimizer_class=torch.optim.Adam,

                 soft_target_tau=1e-2,
                 target_update_period=1,
                 plotter=None,
                 render_eval_paths=False,

                 use_automatic_entropy_tuning=True,
                 target_entropy=None
                 ):

        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy

            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()

            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr
            )

        self.qf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr
        )

        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )

        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr
        )

        self.intrinsic_reward_calculator = intrinsic_reward_calculator

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0

    def train(self, data: td.TransitionModeMapping):
        """
        Args:
            data       : TransitionModeMapping consisting of (N, data_dim, S) data
        """
        seq_dim = -1
        data_dim = -2
        batch_dim = 0

        batch_size = data.obs.size(batch_dim)
        seq_len = data.obs.size(seq_dim)

        data = data.transpose(batch_dim, seq_dim, data_dim)
        data = td.TransitionModeMappingTorch(**self_sup_conversion.from_numpy(data))
        data_dim = -1
        seq_dim = -2

        # Reward (Normalize loss values?)
        intrinsic_rewards = self.intrinsic_reward_calculator.mode_likely_based_rewards(
            obs_seq=data.obs,
            action_seq=data.action,
            skill_gt=data.mode
        )

        # Train SAC
        stacked_data = data.reshape(batch_size * seq_len, -1)
        stacked_rewards = intrinsic_rewards.reshape(batch_size * seq_len, -1)
        self._train_sac(
            batch=stacked_data,
            intrinsic_rewards=stacked_rewards
        )

        self._n_train_steps_total += 1

    def _train_sac(self,
                   batch: td.TransitionModeMappingTorch,
                   intrinsic_rewards: torch.Tensor):
        """
        Args:
            batch                 : (N * S, data_dim) TransitionModeMappingTorch object
            intrinsic_rewards     : (N * S, 1) reward tensor
        """
        batch_dim = 0
        data_dim = -1
        assert batch.size_first_dim == intrinsic_rewards.size(batch_dim)

        obs = batch.obs
        action = batch.action
        next_obs = batch.next_obs
        terminal = batch.terminal
        mode = batch.mode
        reward = intrinsic_rewards

        assert obs.size(data_dim) \
            == next_obs.size(data_dim) \
            == self.env.observation_space.shape[0]
        assert action.size(data_dim) == self.env.action_space.shape[0]
        assert reward.size(data_dim) == terminal.size(data_dim) == 1

        """
        Policy and Alpha Loss
        """
        policy_ret_mapping = self.policy(
            obs=obs,
            skill_vec=mode,
            reparameterize=True,
            return_lob_prob=True
        )
        # just to make auto complete work
        policy_ret_mapping = td.ForwardReturnMapping(**policy_ret_mapping)
        log_pi = policy_ret_mapping.log_prob

        obs_skills = torch.cat((obs, mode), dim=1)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs_skills, policy_ret_mapping.action),
            self.qf2(obs_skills, policy_ret_mapping.action),
        )

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs_skills, action)
        q2_pred = self.qf2(obs_skills, action)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_policy_ret_mapping = self.policy(
            next_obs,
            skill_vec=mode,
            reparameterize=True,
            return_log_prob=True,
        )
        new_policy_ret_mapping = td.ForwardReturnMapping(**new_policy_ret_mapping)
        new_log_pi = new_policy_ret_mapping.log_prob

        next_obs_skills = torch.cat((next_obs, mode), dim=1)
        target_q_values = torch.min(
            self.target_qf1(next_obs_skills, new_policy_ret_mapping.action),
            self.target_qf2(next_obs_skills, new_policy_ret_mapping.action),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * reward + (torch.ones_like(terminal) - terminal) *\
            self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

    @property
    def networks(self) -> Dict[str, nn.Module]:
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )
