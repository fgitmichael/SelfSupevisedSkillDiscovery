import torch
import math
import numpy as np
from torch.nn import functional as F
from operator import itemgetter

from diayn_rnn_seq_rnn_stepwise_classifier.trainer.diayn_step_wise_and_seq_wise_trainer \
    import DIAYNStepWiseSeqWiseRnnTrainer

import self_supervised.utils.my_pytorch_util as my_ptu


class DIAYNAlgoStepwiseSeqwiseRevisedTrainer(DIAYNStepWiseSeqWiseRnnTrainer):
    @property
    def num_skills(self):
        return self.df.output_size

    def _df_loss_seq(self,
                     d_pred_seq,
                     skills_and_id_dict):
        """
        Args:
            d_pred_seq          : (N, num_skills)
            skills_and_id_dict
                skills          : (N, S, skill_dim)
                ids             : (N, S, 1)
        Return:
            df_loss_seq         : scalar tensor
            z_hat_seq           : (N)
            pred_z_seq          : (N)
        """
        skills = skills_and_id_dict['skills']
        skills_id = skills_and_id_dict['ids']

        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = skills.size(batch_dim)
        seq_len = skills.size(seq_dim)
        skill_dim = skills.size(data_dim)

        z_hat_seq = skills_id[:, 0, :].squeeze()
        assert z_hat_seq.shape == torch.Size((batch_size,))
        assert d_pred_seq.shape == torch.Size(
            (batch_size,
             self.num_skills)
        )

        d_pred_log_softmax_seq = F.log_softmax(d_pred_seq, data_dim)
        pred_z_seq = torch.argmax(d_pred_log_softmax_seq, dim=data_dim)

        df_seq_loss = self.df_criterion(
            d_pred_seq,
            z_hat_seq
        )

        return dict(
            df_loss=df_seq_loss,
            pred_z=pred_z_seq,
            z_hat=z_hat_seq
        )

    def _df_loss_step_rewards(
            self,
            d_pred_step,
            skills_and_id_dict,
    ):
        """
        Args:
            d_pred_step         : (N, S, num_skills)
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
        skills_id = skills_and_id_dict['ids']

        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = skills.size(batch_dim)
        seq_len = skills.size(seq_dim)
        skill_dim = skills.size(data_dim)

        assert d_pred_step.shape == torch.Size(
            (batch_size,
             seq_len,
             self.num_skills)
        )

        assert skills_id.shape == torch.Size((batch_size, seq_len, 1))
        z_hat_step = skills_id[..., 0]
        assert z_hat_step.shape == torch.Size(
            (batch_size,
             seq_len))

        d_pred_log_softmax_step = F.log_softmax(d_pred_step, dim=data_dim)
        pred_z_step = torch.argmax(d_pred_log_softmax_step, dim=data_dim)

        # Rewards
        b, s = torch.meshgrid(
            [torch.arange(batch_size),
             torch.arange(seq_len)]
        )

        assert b.shape == s.shape \
               == z_hat_step.shape \
               == torch.Size((batch_size, seq_len))
        rewards = d_pred_log_softmax_step[b, s, z_hat_step] \
                  - math.log(1/self.policy.skill_dim)
        assert rewards.shape == b.shape
        rewards = rewards.unsqueeze(dim=data_dim)
        assert rewards.shape == torch.Size((batch_size, seq_len, 1))

        # Loss
        assert my_ptu.tensor_equality(
            d_pred_step.view(
                batch_size * seq_len,
                self.num_skills
            )[:seq_len],
            d_pred_step[0]
        )
        assert my_ptu.tensor_equality(
            z_hat_step.view(batch_size * seq_len)[:seq_len],
            z_hat_step[0]
        )
        df_step_loss = self.df_criterion(
            d_pred_step.view(
                batch_size * seq_len,
                self.num_skills
            ),
            z_hat_step.view(batch_size * seq_len)
        )

        return dict(
            df_loss=df_step_loss,
            rewards=rewards,
            pred_z=pred_z_step,
            z_hat=z_hat_step
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
        classified_steps, classified_seqs = self.df(next_obs,
                                                    train=True)

        ret_dict_seq = self._df_loss_seq(
            d_pred_seq=classified_seqs,
            skills_and_id_dict=skills_and_id_dict
        )

        df_loss_seq, \
        pred_z_seq, \
        z_hat_seq = itemgetter(
            'df_loss',
            'pred_z',
            'z_hat')(ret_dict_seq)

        ret_dict_step = self._df_loss_step_rewards(
            d_pred_step=classified_steps,
            skills_and_id_dict=skills_and_id_dict
        )

        df_loss_step, \
        pred_z_step, \
        z_hat_step, \
        rewards = itemgetter(
            'df_loss',
            'pred_z',
            'z_hat',
            'rewards')(ret_dict_step)

        return dict(
            df_loss = dict(
                seq=df_loss_seq,
                step=df_loss_step
            ),
            rewards=rewards,
            pred_z = dict(
                seq=pred_z_seq,
                step=pred_z_step
            ),
            z_hat = dict(
                seq=z_hat_seq,
                step=z_hat_step
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