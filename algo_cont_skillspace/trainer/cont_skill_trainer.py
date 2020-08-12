import torch
from torch import nn
from torch import distributions as torch_dist
from itertools import chain
import math
import numpy as np
from torch.nn import functional as F
from operator import itemgetter

from diayn_seq_code_revised.trainer.trainer_seqwise_stepwise_revised import \
    DIAYNAlgoStepwiseSeqwiseRevisedTrainer
from diayn_seq_code_revised.networks.my_gaussian import ConstantGaussianMultiDim

from algo_cont_skillspace.utils.info_loss import InfoLoss
from algo_cont_skillspace.networks.rnn_vae_classifier import RnnVaeClassifierContSkills

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict


class ContSkillTrainer(DIAYNAlgoStepwiseSeqwiseRevisedTrainer):

    def __init__(self,
                 *args,
                 skill_prior_dist: ConstantGaussianMultiDim,
                 loss_fun: InfoLoss.loss,
                 optimizer_class=torch.optim.Adam,
                 df_lr=1e-3,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.skill_prior = skill_prior_dist
        self.loss_fun = loss_fun

        # Overwrite Criterion
        self.df_criterion = nn.MSELoss()

        # Overwrite Optimizer
        assert isinstance(self.df, RnnVaeClassifierContSkills)
        self.df_optimizer_step = optimizer_class(
            chain(
                self.df.classifier.parameters(),
                self.df.feature_decoder.parameters(),
            ),
            lr=df_lr
        )

    @property
    def num_skills(self):
        raise NotImplementedError('Continuous-skills-case: infinite skills')

    def _df_loss_intrinsic_reward(self,
                                  skills,
                                  next_obs):
        """
        Args:
            skills              : (N, S, skill_dim)
            next_obs            : (N, S, obs_dim)
        Return:
            df_loss             : scalar tensor
            rewards             : (N, S, 1)
            log_dict            : dict
        """
        df_ret_dict = self.df(
            next_obs,
            train=True
        )

        classified_steps, \
        feature_recon_dist, \
        classified_seqs, \
        hidden_features_seq = itemgetter(
            'classified_steps',
            'feature_recon_dist',
            'classified_seqs',
            'hidden_features_seq')(df_ret_dict)

        # Sequence Classification Loss
        ret_dict_seq = self._df_loss_seq(
            d_pred_per_seq=classified_seqs,
            skills=skills
        )
        df_loss_seq = itemgetter(
            'df_loss',
        )(ret_dict_seq)

        # Step Loss and rewards
        loss_calc_values = dict(
            hidden_feature_seq=hidden_features_seq,
            recon_feature_seq=feature_recon_dist,
            post_skills=classified_steps
        )
        ret_dict_step = self._df_loss_step_rewards(
            loss_calc_values=loss_calc_values,
            skills=skills
        )
        df_loss_step, \
        rewards, \
        log_dict_df_step = itemgetter(
            'df_loss',
            'rewards',
            'log_dict')(ret_dict_step)

        return dict(
            df_loss=dict(
                seq=df_loss_seq,
                step=df_loss_step,
            ),
            rewards=rewards,
            log_dict=log_dict_df_step
        )

    def _df_loss_step_rewards(
            self,
            loss_calc_values: dict,
            skills: torch.Tensor,
    ):
        """
        Args:
            loss_calc_values
                hidden_feature_seq  : (N, S, hidden_size_rnn)
                recon_feature_seq   : (N, S, hidden_size_rnn) distributions
                post_skills         : (N, S, skill_dim) distributions
            skills                  : (N, S, skill_dim)

        Return:
            df_loss                 : scalar tensor
            rewards                 : (N, S, 1)
            log_dict
                kld                 : scalar tensor
                mmd                 : scalar tensor
                mse                 : scalar tensor
                kld_info            : scalar tensor
                mmd_info            : scalar tensor
                loss_latent         : scalar tensor
                loss_data           : scalar tensor
                info_loss           : scalar tensor
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = skills.size(batch_dim)
        seq_len = skills.size(seq_dim)
        skill_dim = skills.size(data_dim)

        hidden_feature_seq = loss_calc_values['hidden_feature_seq']
        recon_feature_seq = loss_calc_values['recon_feature_seq']
        post_skills = loss_calc_values['post_skills']

        assert hidden_feature_seq.shape == torch.Size(
            (batch_size,
             seq_len,
             skill_dim))
        assert post_skills.batch_shape == skills.shape
        assert hidden_feature_seq.shape == recon_feature_seq.batch_shape

        # Rewards
        rewards = post_skills.log_prob(skills)
        rewards = torch.sum(rewards, dim=data_dim, keepdim=True)
        assert rewards.shape == torch.Size((batch_size, seq_len, 1))

        # Loss
        pri = dict(
            dist=self.skill_prior(hidden_feature_seq),
            sample=self.skill_prior.sample(),
        )
        post = dict(
            dist=post_skills,
            sample=post_skills.rsample()
        )
        recon = dict(
            dist=recon_feature_seq,
            sample=recon_feature_seq.loc,
        )
        info_loss, log_dict = self.loss_fun(
            pri=pri,
            post=post,
            recon=recon,
            data=hidden_feature_seq
        )

        return dict(
            df_loss=info_loss,
            rewards=rewards,
            df_loss_logging=log_dict
        )

    def _df_loss_seq(self,
                     d_pred_per_seq,
                     skills):
        """
        Args:
            d_pred_per_seq      : (N, skill_dim) predicted skills per seq
            skills              : (N, S, skill_dim) ground truth skills seq
        Return:
            df_loss_seq         : scalar tensor
            pred_skill          : (N, skill_dim)
            skill_gt            : (N, skill_dim)
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = skills.size(batch_dim)
        seq_len = skills.size(seq_dim)
        skill_dim = skills.size(data_dim)

        assert d_pred_per_seq.batch_shape == torch.Size(
            (batch_size,
             skill_dim)
        )

        skills_per_seq_gt = skills[:, 0, :]
        assert skills_per_seq_gt.shape == torch.Size((batch_size, skill_dim))
        assert torch.stack([skills_per_seq_gt] * seq_len, dim=seq_dim) == skills

        # Apply MSE Loss
        df_seq_loss = self.df_criterion(
            d_pred_per_seq,
            skills_per_seq_gt
        )

        return dict(
            df_loss=df_seq_loss,
        )

    def train_from_torch(self, batch):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1

        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        skills = batch['skills']
        skills_id = batch['skills_id'].long()

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
        log_dict = itemgetter(
            'df_loss',
            'rewards',
            'log_dict'
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
            log_dict=log_dict,
            pred_z=None,
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
                    log_dict,
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
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            for key, el in log_dict.items():
                self.eval_statistics[key] = ptu.get_numpy(el)

            self.eval_statistics['Intrinsic Rewards'] = \
                np.mean(ptu.get_numpy(rewards))
            self.eval_statistics['DF Loss Seq'] = \
                np.mean(ptu.get_numpy(df_loss['seq']))
            self.eval_statistics['DF Loss Step'] = \
                np.mean(ptu.get_numpy(df_loss['step']))
            self.eval_statistics['QF1 Loss'] = \
                np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = \
                np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = \
                np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'D Predictions Step',
                ptu.get_numpy(pred_z['step']),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'D Predictions Seq',
                ptu.get_numpy(pred_z['seq']),
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
