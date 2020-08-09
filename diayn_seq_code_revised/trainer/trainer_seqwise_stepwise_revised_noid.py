import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import itemgetter

from diayn_seq_code_revised.trainer.trainer_seqwise_stepwise_revised import \
    DIAYNAlgoStepwiseSeqwiseRevisedTrainer


class DIAYNAlgoStepwiseSeqwiseRevisedNoidTrainer(DIAYNAlgoStepwiseSeqwiseRevisedTrainer):

    def __init__(self, *args, num_skills, **kwargs):
        super().__init__(*args, **kwargs)

        self._num_skills = num_skills

        # Overwrite Criterion
        self.df_criterion = nn.MSELoss()

    @property
    def num_skills(self):
        return self._num_skills

    def _df_loss_seq(self,
                     d_pred_seq,
                     skills_and_id_dict):
        """
        Args:
            d_pred_seq          : (N, skill_dim)
            skills_and_id_dict
                skills          : (N, S, skill_dim)
                ids             : (N, S, 1)
        Return:
            df_loss_seq         : scalar tensor
            z_hat_seq           : (N)
            pred_z_seq          : (N)
        """
        skills = skills_and_id_dict['skills']

        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = skills.size(batch_dim)
        seq_len = skills.size(seq_dim)
        skill_dim = skills.size(data_dim)

        skills_per_seq_gt = skills[:, 0, :]
        assert skills_per_seq_gt.shape == torch.Size((batch_size, skill_dim))
        assert d_pred_seq.shape == torch.Size(
            (batch_size,
             skill_dim)
        )

        df_seq_loss = self.df_criterion(
            d_pred_seq,
            skills_per_seq_gt
        )

        return dict(
            df_loss=df_seq_loss,
            pred_skill=skills_per_seq_gt,
            skill_gt=skills_per_seq_gt
        )

    def _df_loss_step_rewards(
            self,
            d_pred_step_dist,
            skills_and_id_dict,
    ):
        """
        Args:
            d_pred_step_dist    : (N, S, skill_dim) distribution
            skills_and_id_dict
                skills          : (N, S, skill_dim)
                ids             : (N, S, 1)
        Return:
            df_loss_step        : scalar tensor
            rewards             : (N, S, 1)
            z_hat_step          : (N, S)
            pred_z_step         : (N, S)
        """
        skills = skills_and_id_dict['skills']

        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = skills.size(batch_dim)
        seq_len = skills.size(seq_dim)
        skill_dim = skills.size(data_dim)

        assert d_pred_step_dist.loc.shape == torch.Size(
            (batch_size,
             seq_len,
             skill_dim))

        # Rewards
        rewards = d_pred_step_dist.log_prob(skills)
        assert rewards.shape == torch.Size((batch_size, seq_len, 1))

        # Loss
        loss = -rewards.sum()

        return dict(
            df_loss=loss,
            rewards=rewards,
            pred_skill=d_pred_step_dist.loc,
            skill_gt=skills
        )

    def _df_loss_intrinsic_reward(self,
                                  skills_and_id_dict,
                                  next_obs):
        """
        Args:
            skills_and_id_dict
                skills          : (N, S, skill_dim)
                ids             : (N, S, 1)
            next_obs            : (N, S, obs_dim)
        Return:
            df_loss             : (N, 1)
            rewards             : (N, S, 1)
            pred_z              : (N, S) skill predicted skill ID's
            z_hat               : (N, S) skill ground truth skill ID's
        """
        classfied_step_dists, classified_seqs = self.df(next_obs,
                                                        train=True)

        ret_dict_seq = self._df_loss_seq(
            d_pred_seq=classified_seqs,
            skills_and_id_dict=skills_and_id_dict
        )

        df_loss_seq, \
        pred_skill_seq, \
        skills_gt_seq = itemgetter(
            'df_loss',
            'pred_skill',
            'skill_gt')(ret_dict_seq)

        ret_dict_step = self._df_loss_step_rewards(
            d_pred_step_dist=classfied_step_dists,
            skills_and_id_dict=skills_and_id_dict
        )

        df_loss_step, \
        pred_skill_step, \
        skills_gt_step, \
        rewards = itemgetter(
            'df_loss',
            'pred_skill',
            'skill_gt',
            'rewards')(ret_dict_step)

        return dict(
            df_loss=dict(
                seq=df_loss_seq,
                step=df_loss_step
            ),
            rewards=rewards,
            pred_z=dict(
                seq=pred_skill_seq,
                step=pred_skill_step
            ),
            skill_gt=dict(
                seq=skills_gt_seq,
                step=skills_gt_step
            )
        )

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
            skills_and_id_dict=dict(
                skills=skills,
                ids=skills_id
            ),
            next_obs=next_obs
        )

        df_loss, \
        rewards, \
        pred_skill, \
        skill_gt = itemgetter(
            'df_loss',
            'rewards',
            'pred_skill',
            'skill_gt',
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
            z_hat=skill_gt,
            pred_z=pred_skill,
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

    def _calc_df_accuracy(self, z_hat, pred_z):
        """
        Args:
            z_hat
                step                    : (N, S, skill_dim)
                seq                     : (N, skill_dim)
            pred_z
                step                    : (N, S, skill_dim)
                seq                     : (N, skill_dim
        """
        assert z_hat['step'].shape == pred_z['step'].shape
        assert z_hat['seq'].shape == pred_z['seq'].shape

        df_accuracy_step = F.mse_loss(z_hat['step'], pred_z['step'])
        df_accuracy_seq = F.mse_loss(z_hat['seq'], pred_z['seq'])

        return dict(
            step=df_accuracy_step,
            seq=df_accuracy_seq
        )
