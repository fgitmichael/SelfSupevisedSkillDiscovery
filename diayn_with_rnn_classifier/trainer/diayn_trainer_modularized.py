import math
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from operator import itemgetter

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.diayn.diayn import DIAYNTrainer
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class DIAYNTrainerModularized(DIAYNTrainer):

    def train_from_torch(self, batch):
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        skills = batch['skills']


        """
        DF Loss and Intrinsic Reward
        """
        df_ret_dict = self._df_loss_intrinsic_reward(
            skills=skills,
            next_obs=next_obs
        )

        df_loss, \
        rewards, \
        pred_z, \
        z_hat = itemgetter(
            'df_loss',
            'rewards',
            'pred_z',
            'z_hat',
        )(df_ret_dict)


        """
        Policy and Alpha Loss
        """
        policy_ret_dict = self._policy_alpha_loss(
            obs=obs,
            skills=skills
        )

        policy_loss, \
        alpha_loss, \
        alpha, \
        q_new_actions, \
        policy_mean, \
        policy_log_std, \
        log_pi, \
        obs_skills = itemgetter(
            'policy_loss',
            'alpha_loss',
            'alpha',
            'q_new_actions',
            'policy_mean',
            'policy_log_std',
            'log_pi',
            'obs_skills'
        )(policy_ret_dict)


        """
        QF Loss
        """
        qf_ret_dict = self._qf_loss(
            actions=actions,
            next_obs=next_obs,
            alpha=alpha,
            rewards=rewards,
            terminals=terminals,
            skills=skills,
            obs_skills=obs_skills
        )

        qf1_loss, \
        qf2_loss, \
        q1_pred, \
        q2_pred, \
        q_target = itemgetter(
            'qf1_loss',
            'qf2_loss',
            'q1_pred',
            'q2_pred',
            'q_target'
        )(qf_ret_dict)


        """
        Update networks
        """
        self._update_networks(
            df_loss=df_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            policy_loss=policy_loss
        )


        """
        Soft Updates
        """
        self._soft_updates()


        """
        Save some statistics for eval
        """
        self._save_stats(
            z_hat=z_hat,
            pred_z=pred_z,
            log_pi=log_pi,
            q_new_actions=q_new_actions,
            rewards=rewards,
            df_loss=df_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            q1_pred=q1_pred,
            q2_pred=q2_pred,
            q_target=q_target,
            policy_mean=policy_mean,
            policy_log_std=policy_log_std,
            alpha=alpha,
            alpha_loss=alpha_loss
        )

    def _df_loss_intrinsic_reward(self,
                                  skills,
                                  next_obs):
        z_hat = torch.argmax(skills, dim=1)
        d_pred = self.df(next_obs)
        d_pred_log_softmax = F.log_softmax(d_pred, 1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        rewards = d_pred_log_softmax[torch.arange(d_pred.shape[0]), z_hat] - math.log(1/self.policy.skill_dim)
        rewards = rewards.reshape(-1, 1)
        df_loss = self.df_criterion(d_pred, z_hat)

        return dict(
            df_loss=df_loss,
            rewards=rewards,
            pred_z=pred_z,
            z_hat=z_hat
        )

    def _policy_alpha_loss(self,
                           obs,
                           skills,
                           ):
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

        return dict(
            policy_loss=policy_loss,
            alpha_loss=alpha_loss,
            alpha=alpha,
            q_new_actions=q_new_actions,
            policy_mean=policy_mean,
            policy_log_std=policy_log_std,
            log_pi=log_pi,
            obs_skills=obs_skills,
        )

    def _qf_loss(self,
                 actions,
                 next_obs,
                 alpha,
                 rewards,
                 terminals,
                 skills,
                 obs_skills,
                 ):
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

        return dict(
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            q1_pred=q1_pred,
            q2_pred=q2_pred,
            q_target=q_target,
        )

    def _update_networks(self,
                         df_loss,
                         qf1_loss,
                         qf2_loss,
                         policy_loss):
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

    def _soft_updates(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

    def _save_stats(self,
                    z_hat,
                    pred_z,
                    log_pi,
                    q_new_actions,
                    rewards,
                    df_loss,
                    qf1_loss,
                    qf2_loss,
                    q1_pred,
                    q2_pred,
                    q_target,
                    policy_mean,
                    policy_log_std,
                    alpha,
                    alpha_loss
                    ):
        """
        Save some statistics for eval
        """
        df_accuracy = torch.sum(torch.eq(z_hat, pred_z.reshape(1, list(pred_z.size())[0])[0])).float()/list(pred_z.size())[0]

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['Intrinsic Rewards'] = np.mean(ptu.get_numpy(rewards))
            self.eval_statistics['DF Loss'] = np.mean(ptu.get_numpy(df_loss))
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


