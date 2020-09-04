from operator import itemgetter
from itertools import chain

import torch

from seqwise_cont_skillspace.trainer.cont_skillspace_seqwise_trainer import \
    ContSkillTrainerSeqwiseStepwise


class ContSkillTrainerSeqwiseStepwiseHighdimusingvae(ContSkillTrainerSeqwiseStepwise):

    def create_optimizer_seq(self, optimizer_class, df_lr):
        return optimizer_class(
            chain(
                self.df.classifier_seq.parameters(),
                self.df.rnn.parameters(),
            ),
            lr=df_lr,
        )

    def create_optimizer_step(self, optimizer_class, df_lr):
        return optimizer_class(
            self.df.classifier_step.parameters(),
            lr=df_lr,
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
                skill_recon         : (N * S, skill_recon) dist and sample
                latent_post         : (N * S, skill_dim) dist and sample
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
        hidden_feature_seq, \
        skill_recon, \
        latent_post = itemgetter(
            'hidden_feature_seq',
            'skill_recon',
            'latent_post')(loss_calc_values)

        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size, seq_len, skill_dim = skills.shape
        assert skill_recon['dist'].batch_shape[:-1] \
               == latent_post['dist'].batch_shape[:-1] \
               == torch.Size((hidden_feature_seq.size(batch_dim)
                              * hidden_feature_seq.size(seq_dim),)) \
               == torch.Size((skills.size(batch_dim)
                              * hidden_feature_seq.size(seq_dim),))

        skills = skills.reshape(batch_size * seq_len, skill_dim)
        assert skills.shape == skill_recon['dist'].batch_shape


        # Rewards
        rewards = skill_recon['dist'].log_prob(skills)
        rewards = torch.sum(rewards, dim=data_dim, keepdim=True)
        assert rewards.shape == torch.Size((batch_size * seq_len, 1))

        # Reshaping to (N * S, D) format
        hidden_seq_stacked_detached = hidden_feature_seq.reshape(
            batch_size * seq_len,
            hidden_feature_seq.size(data_dim)).detach()
        pri_dist = self.reshape_normal(self.skill_prior(hidden_feature_seq))
        assert len(pri_dist.batch_shape) == 2
        pri = dict(
            dist=pri_dist,
            sample=pri_dist.sample()
        )

        # Loss Calculation
        info_loss, log_dict = self.loss_fun(
            pri=pri,
            post=latent_post,
            recon=skill_recon,
            data=skills.detach(),
        )

        return dict(
            df_loss=info_loss,
            rewards=rewards,
            log_dict=log_dict,
        )

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

        skill_recon, \
        latent_post, \
        classified_seqs, \
        hidden_features_seq = itemgetter(
            'skill_recon',
            'latent_post',
            'classified_seqs',
            'hidden_features_seq')(df_ret_dict)

        # Sequence Classification Loss
        ret_dict_seq = self._df_loss_seq(
            pred_skills_seq=classified_seqs,
            skills=skills
        )
        df_loss_seq = itemgetter(
            'df_loss',
        )(ret_dict_seq)

        # Step Loss and rewards
        loss_calc_values = dict(
            hidden_feature_seq=hidden_features_seq,
            skill_recon=skill_recon,
            latent_post=latent_post,
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

    def train_from_torch(self, batch):
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

