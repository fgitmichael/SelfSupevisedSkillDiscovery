import torch
from torch.nn import functional as F
import numpy as np
from operator import itemgetter

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict

from diayn_original_cont.networks.vae_regressor import VaeRegressor
from diayn_original_cont.trainer.info_loss_min_vae import InfoLossLatentGuided

from diayn_with_rnn_classifier.trainer.diayn_trainer_modularized import \
    DIAYNTrainerModularized


class DIAYNTrainerCont(DIAYNTrainerModularized):

    def __init__(self,
                 *args,
                 info_loss_fun: InfoLossLatentGuided.loss,
                 **kwargs):
        super(DIAYNTrainerCont, self).__init__(*args, **kwargs)
        self.info_loss_fun = info_loss_fun

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
        pred_skill, \
        log_dict = itemgetter(
            'df_loss',
            'rewards',
            'pred_skill',
            'log_dict',
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
            dummy=None,
            df_accuracy=F.mse_loss(pred_skill, skills),
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

    def _save_stats(self,
                    dummy,
                    df_accuracy,
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
        Args:
            dummy           : for avoiding changes of signature
        """
        """
        Save some statistics for eval
        """
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

    def _df_loss_intrinsic_reward(self,
                                  skills,
                                  next_obs):
        """
        Args:
            next_obs            : (N, obs_dim)
            skills              : (N, skill_dim)
        """
        assert len(next_obs.shape) == len(skills.shape) == 2
        batch_dim = 0
        data_dim = -1

        z_hat = skills
        train_dict = self.df(next_obs, train=True)

        latent_post_dist = train_dict['post']['dist']
        assert latent_post_dist.batch_shape == skills.shape
        rewards = torch.sum(
            latent_post_dist.log_prob(skills),
            dim=data_dim,
            keepdim=True,
        )
        assert rewards.shape == torch.Size((next_obs.size(0), 1))

        vae_ret_dict = self.df(next_obs, train=True)
        info_loss, log_dict = self.info_loss_fun(
            **vae_ret_dict,
            data=next_obs,
            latent_guide=skills,
        )

        return dict(
            df_loss=info_loss,
            rewards=rewards,
            pred_skill=vae_ret_dict['post']['dist'].loc,
            log_dict=log_dict,
        )
