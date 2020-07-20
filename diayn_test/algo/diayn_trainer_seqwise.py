import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
from typing import Dict

from self_supervised.base.trainer.trainer_base import MyTrainerBaseClass

from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.policy.skill_policy import SkillTanhGaussianPolicy
import self_supervised.utils.conversion as self_sup_conversion
import self_supervised.utils.typed_dicts as td

from rlkit.torch.networks import FlattenMlp
import rlkit.torch.pytorch_util as ptu

class DiaynTrainerSeqwise(MyTrainerBaseClass):

    def __init__(self,
                 env: NormalizedBoxEnvWrapper,
                 policy: SkillTanhGaussianPolicy,
                 qf1: FlattenMlp,
                 qf2: FlattenMlp,
                 target_qf1: FlattenMlp,
                 target_qf2: FlattenMlp,
                 df: FlattenMlp,

                 discount=0.99,
                 reward_scale=1.0,

                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 df_lr=1e-3,
                 optimizer_class=torch.optim.Adam,

                 soft_target_tau=1e-2,
                 target_update_period=1,
                 plotter=None,
                 render_eval_paths=False,

                 use_automatic_entropy_tuning=True,
                 target_entropy=None):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.df = df
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.df_criterion = nn.CrossEntropyLoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.df_optimizer = optimizer_class(
            self.df.parameters(),
            lr=df_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.skill_grid = self.get_grid()

    def train(self, data: td.TransitonModeMappingDiscreteSkills):
        """
        Args:
            data      : (N, data_dim, S)
        """
        seq_dim = -1
        data_dim = -2
        batch_dim = 0

        batch_size = data.obs.shape[batch_dim]
        seq_len = data.obs.shape[seq_dim]

        data = data.transpose(batch_dim, seq_dim, data_dim)
        data = td.TransitonModeMappingDiscreteSkills(**self_sup_conversion.from_numpy(data))
        data_dim = -1
        seq_dim = -2

        obs = ptu.from_numpy(data.obs)
        mode = ptu.from_numpy(data.mode)
        next_obs = ptu.from_numpy(data.next_obs)
        action = ptu.from_numpy(data.action)
        terminal = ptu.from_numpy(data.terminal)
        skill_id = ptu.from_numpy(data.skill_id)

        df_loss, reward = self.df_loss_rewards(
            next_obs=next_obs,
            skill_id=skill_id
        )

        """
        Policy and Alpha Loss
        """
        policy_ret_mapping = self.policy(
            obs=obs,
            skill_vec=mode,
            reparameterize=True,
            return_log_prob=True
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

    def df_loss_rewards(self,
                        skill_id,
                        next_obs
                        ):
        """
        Args:
            data       : (N, S, data_dim)
        """
        data_dim = -1

        z_hat = skill_id
        d_pred = self.df(next_obs)
        assert d_pred.shape[:data_dim] == next_obs.shape[:data_dim]
        assert d_pred.shape[data_dim] == self.skill_grid.size(0)

        d_pred_log_softmax = F.log_softmax(d_pred, dim=data_dim)
        assert d_pred_log_softmax.shape == d_pred.shape
        _, pred_z = torch.max(d_pred_log_softmax, dim=data_dim, keepdim=True)
        assert pred_z.shape[:data_dim] == d_pred_log_softmax.shape[:data_dim]
        assert pred_z.size(data_dim) == 1

        rewards = d_pred_log_softmax[:, :, z_hat] - math.log(1/self.policy.skill_dim)
        assert rewards.shape[:data_dim] == next_obs.shape[:data_dim]
        assert rewards.size(data_dim) == 1

        df_loss = self.df_criterion(d_pred.reshape(-1, 1), z_hat)

        return df_loss, rewards

    def get_grid(self):
        # Hard coded for testing
        radius1 = 0.75
        radius2 = 1.
        radius3 = 1.38
        grid = np.array([
            [0., 0.],
            [radius1, 0.],
            [0., radius1],
            [-radius1, 0.],
            [0, -radius1],
            [radius2, radius2],
            [-radius2, radius2],
            [radius2, -radius2],
            [-radius2, -radius2],
            [0, radius3]
        ], dtype=np.float)

        grid = ptu.from_numpy(grid)

        return grid

    def end_epoch(self, epoch):
        pass

    def get_diagnostics(self):
        pass

    def get_snapshot(self):
        pass

    @property
    def networks(self) -> Dict[str, nn.Module]:
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            df=self.df
        )
