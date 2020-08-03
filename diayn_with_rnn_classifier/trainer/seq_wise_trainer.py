import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import math
from operator import itemgetter

from diayn_with_rnn_classifier.trainer.diayn_trainer_modularized import \
    DIAYNTrainerModularized

import self_supervised.utils.my_pytorch_util as my_ptu

from rlkit.core.eval_util import create_stats_ordered_dict
import rlkit.torch.pytorch_util as ptu

class DIAYNTrainerSeqWise(DIAYNTrainerModularized):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_from_torch(self, batch):
        """
        Args:
            data          : (N, S, data_dim)
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1

        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        skills = batch['skills']

        assert terminals.shape[:-1] \
            == obs.shape[:-1] \
            == actions.shape[:-1] \
            == next_obs.shape[:-1] \
            == skills.shape[:-1]
        assert obs.size(data_dim) == next_obs.size(data_dim)
        assert skills.size(data_dim) == self.policy.skill_dim
        assert terminals.size(data_dim) == 1
        batch_size = next_obs.size(batch_dim)
        seq_len = next_obs.size(seq_dim)
        obs_dim = obs.size(data_dim)
        action_dim = actions.size(data_dim)

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

        num_transitions = batch_size * seq_len
        assert torch.all(
            torch.eq(obs[0, 0, :],
                     obs.view(num_transitions, obs_dim)[0, :])
        )
        terminals = terminals.view(num_transitions, 1)
        obs = obs.view(num_transitions, obs_dim)
        next_obs = next_obs.view(num_transitions, obs_dim)
        actions = actions.view(num_transitions, action_dim)
        skills = skills.view(num_transitions, self.policy.skill_dim)
        assert torch.all(
            torch.eq(
                rewards[0, 0, :],
                rewards.view(num_transitions, 1)[0, :])
        )
        rewards = rewards.view(num_transitions, 1)
        assert torch.all(
            torch.eq(pred_z[0, :],
                     pred_z.view(num_transitions, 1)[:seq_len, :].squeeze())
        )
        pred_z = pred_z.view(num_transitions, 1)
        z_hat = z_hat.view(-1, 1)


        # No changes from here on
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
        """
        Args:
            skills            : (N, S, skill_dim)
            next_obs          : (N, S, obs_dim)
        Return:
            df_loss           : (N, 1)
            rewards           : (N, S, 1)
            pred_z            : (N, S) skill predicted skill ID's
            z_hat             : (N, 1) skill ground truth skill ID's
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = next_obs.size(batch_dim)
        seq_len = next_obs.size(seq_dim)
        obs_dim = next_obs.size(data_dim)

        skills_per_seq = skills[:, 0, :]
        next_obs_stacked = next_obs.view(batch_size * seq_len, obs_dim)

        z_hat = torch.argmax(skills, dim=data_dim)
        z_hat_oh = skills_per_seq
        d_pred = self.df(next_obs)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=data_dim)
        assert d_pred_log_softmax.shape == torch.Size((batch_size, seq_len, d_pred.size(-1)))
        pred_z = torch.argmax(d_pred_log_softmax, dim=data_dim)

        df_loss = self.df_criterion(
            d_pred.view(-1, 10),
            z_hat.squeeze().view(-1, 1).squeeze())

        b, s = torch.meshgrid([torch.arange(batch_size), torch.arange(seq_len)])
        rewards = d_pred_log_softmax[b, s, z_hat].unsqueeze(dim=data_dim) - \
            math.log(1/self.policy.skill_dim)

        assert rewards.shape[:data_dim] == next_obs.shape[:data_dim]
        assert rewards.size(data_dim) == 1
        assert df_loss.shape == torch.Size(())
        assert pred_z.shape == torch.Size((batch_size, seq_len))
        assert z_hat.shape == torch.Size((batch_size, seq_len))

        return dict(
            df_loss=df_loss,
            rewards=rewards,
            pred_z=pred_z,
            z_hat=z_hat
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
        df_accuracy = torch.sum(torch.eq(z_hat, pred_z)).float()/pred_z.size(0)

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

    @property
    def num_skills(self):
        """
        Add property cause number of skills can change with different
        implementations (i.e. with skills not encoded as one-hot vectors)
        """
        return self.policy.skill_dim
