import torch
import torch.nn as nn
import numpy as np
from typing import Iterable

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import Trainer
from rlkit.torch.networks import FlattenMlp

from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.policy.skill_policy import SkillTanhGaussianPolicy
from self_supervised.utils.typed_dicts import *
from self_supervised.loss.loss_intrin_selfsup import reconstruction_based_rewards
from self_supervised.algo.trainer_mode_latent import \
    ModeLatentTrainer, ModeLatentNetworkWithEncoder
import self_supervised.utils.conversion as self_sup_conversion


class SelfSupTrainer(Trainer):
    def __init__(self,
                 env: NormalizedBoxEnvWrapper,
                 policy: SkillTanhGaussianPolicy,
                 qf1: FlattenMlp,
                 qf2: FlattenMlp,
                 target_qf1: FlattenMlp,
                 target_qf2: FlattenMlp,
                 mode_latent_model: ModeLatentNetworkWithEncoder,

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
        super().__init__()

        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.mode_latent_model = mode_latent_model

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
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0

    def train(self, data: TransitionModeMapping):
        """
        data        : TransitionModeMapping consisting of (N, dim, seq_len) data
        """
        data = self_sup_conversion.from_numpy(data)

        # Reward
        # TODO: Normalize loss values?
        intrinsic_rewards = reconstruction_based_rewards(
            mode_latent_model=self.mode_latent_model,
            obs_seq=data.obs,
            action_seq=data.action,
            skill_seq=data.mode
        )

        # Train SAC
        for idx, transition in \
                enumerate(data.get_transition_mapping().permute(2, 0, 1)):
            self.train_sac(
                batch=transition,
                mode=data.mode,
                intrinsic_rewards=intrinsic_rewards[:, :, idx]

            )

    def train_sac(self, batch: TransitionMappingTorch,
                         mode: torch.Tensor,
                         intrinsic_rewards: torch.Tensor):
        """
        batch               : TransitionModeMapping consisting of  (N, dim) data
        mode                : (N, mode_dim) tensor
        intrinsic_rewards   : (N, 1) tensor
        """
        obs = batch.obs
        actions = batch.action
        next_obs = batch.next_obs
        terminals = batch.terminal
        skills = mode
        rewards = intrinsic_rewards

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, skill_vec=skills, reparameterize=True, return_log_prob=True,
        )
        obs_skills = torch.cat((obs, skills), dim=1)
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
            self.qf1(obs_skills, new_obs_actions),
            self.qf2(obs_skills, new_obs_actions),
        )

        assert type(q_new_actions) \
               == type(log_pi) \
               == torch.Tensor

        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs_skills, actions)
        q2_pred = self.qf2(obs_skills, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, skill_vec=skills, reparameterize=True, return_log_prob=True,
        )
        next_obs_skills = torch.cat((next_obs, skills), dim=1)
        target_q_values = torch.min(
            self.target_qf1(next_obs_skills, new_next_actions),
            self.target_qf2(next_obs_skills, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) *\
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
    def networks(self) -> Iterable[nn.Module]:
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
            mode_latent=self.mode_latent_model,
        )


