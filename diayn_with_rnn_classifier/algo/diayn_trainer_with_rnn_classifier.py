from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import functional as F
import math

from code_slac.utils import update_params as backprop

from rlkit.torch.torch_rl_algorithm import TorchTrainer
import rlkit.torch.pytorch_util as ptu

from diayn_with_rnn_classifier.reward_calculation.reward_calculator \
    import RewardPolicyDiff
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

# Replace df by rnn classfier
class DIAYNTrainerRnnClassifier(TorchTrainer):

    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            df,
            reward_calculator: RewardPolicyDiff,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            df_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.df = df
        self.reward_calculator = reward_calculator
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()
                # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

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
            lr=df_lr
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        """
        terminals           : (N, S, 1) tensor
        obs                 : (N, S, obs_dim) tensor
        actions             : (N, S, action_dim) tensor
        next_obs            : (N, S, obs_dim) tensor
        skills              : (N, S, skill_dim) tensor
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        skills = batch['skills']

        batch_size = obs.size(batch_dim)
        seq_len = obs.size(seq_dim)
        assert obs.shape[:-1] \
               == actions.shape[:-1] \
               == next_obs.shape[:-1] \
               == skills.shape[:-1] \
               == terminals.shape[:-1] \
               == rewards.shape[:-1]

        """
        DF Loss and Intrinsic Reward
        """
        assert skills.shape[:-1] == torch.Size((batch_size, seq_len))
        z_hat = torch.argmax(skills, dim=data_dim)
        assert torch.all(
            torch.stack([z_hat[:, 0]] * z_hat.size(seq_dim), dim=seq_dim) \
                == z_hat)
        d_pred = self.df(next_obs)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=-1)
        pred_z = torch.argmax(d_pred_log_softmax, dim=-1, keepdim=True)
        df_loss = self.df_criterion(d_pred, z_hat)

        rewards = self.reward_calculator.calc_rewards(
            obs_seq=obs,
            action_seq=actions,
            skill_gt_id=z_hat[:, 0, :],
            pred_log_softmax=d_pred_log_softmax
        )

        """
        Stack data for rest of method
        """
        batch_size_stacked = batch_size * seq_len
        assert torch.all(
            torch.eq(obs.view(batch_size_stacked, obs.size(data_dim))[0, :], obs[0, 0, :])
        )
        obs = obs.view(batch_size_stacked, obs.size(data_dim))
        actions = actions.view(batch_size_stacked, actions.size(data_dim))
        next_obs = next_obs.view(batch_size_stacked, next_obs.size(data_dim))
        terminals = terminals.view(batch_size_stacked, terminals.size(data_dim))
        skills = skills.view(batch_size_stacked, skills.size(data_dim))
        rewards = rewards.view(batch_size_stacked, rewards.size(data_dim))
        z_hat = z_hat.view(batch_size_stacked, z_hat.size(data_dim))
        assert d_pred_log_softmax.shape == torch.Size((batch_size, skills.size(data_dim)))
        prediction_log_softmax = torch.stack([d_pred_log_softmax] * seq_len, dim=seq_dim)
        prediction_log_softmax_stacked = prediction_log_softmax.view(
            batch_size_stacked,
            prediction_log_softmax.size(data_dim)
        )
        assert torch.all(
            prediction_log_softmax[0, 0, :] == prediction_log_softmax_stacked[0, :]
        )
        predicted_labels = torch.argmax(prediction_log_softmax,
                                        dim=data_dim,
                                        keepdim=True)
        pred_z = predicted_labels.view(-1, 1)
        assert torch.all(
            pred_z[:seq_len, :] == predicted_labels[0, :, :]
        )

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, skill_vec=skills, reparameterize=True, return_log_prob=True,
        )
        obs_skills = torch.cat((obs, skills), dim=1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
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
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs_skills, actions)
        q2_pred = self.qf2(obs_skills, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, skill_vec = skills, reparameterize=True, return_log_prob=True,
        )
        next_obs_skills = torch.cat((next_obs, skills), dim=1)
        target_q_values = torch.min(
            self.target_qf1(next_obs_skills, new_next_actions),
            self.target_qf2(next_obs_skills, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.df_optimizer.zero_grad()
        df_loss.backward()
        self.df_optimizer.step()

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

        """
        Save some statistics for eval
        """
        # Change from original
        #df_accuracy = torch.sum(torch.eq(labels, pred_z.reshape(1, list(pred_z.size())[0])[0])).float()/list(pred_z.size())[0]
        df_accuracy = torch.sum(
            torch.eq(
                z_hat,
                pred_z.squeeze()
            )).float()/pred_z.size(0)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['Intrinsic Rewards'] = np.mean(ptu.get_numpy(rewards))
            self.eval_statistics['Classfier loss'] = np.mean(ptu.get_numpy(df_loss))
            self.eval_statistics['DF Accuracy'] = np.mean(ptu.get_numpy(df_accuracy))
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'D Predictions',
                ptu.get_numpy(pred_z),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.df
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
            df=self.df
        )


class DIAYNTrainerRnnClassifierExtension(DIAYNTrainerRnnClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_samples_trained = 0

    def train_from_torch(self, batch):
        obs = batch['observations']
        batch_size = obs.shape[0]
        self.num_samples_trained += batch_size

        super().train_from_torch(batch)

        self.eval_statistics.update(create_stats_ordered_dict(
            'Number of Samples used for Training',
            self.num_samples_trained
        ))
