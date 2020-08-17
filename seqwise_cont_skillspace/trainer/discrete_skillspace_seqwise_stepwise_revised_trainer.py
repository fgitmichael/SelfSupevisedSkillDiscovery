import torch
import torch.distributions as torch_dist
from torch import nn
from itertools import chain
from operator import itemgetter

from seqwise_cont_skillspace.trainer.cont_skillspace_seqwise_trainer import \
    ContSkillTrainerSeqwiseStepwise


class DiscreteSkillTrainerSeqwiseStepwise(ContSkillTrainerSeqwiseStepwise):

    def __init__(self,
                 *args,
                 skill_prior_dist,
                 loss_fun,
                 optimizer_class=torch.optim.Adam,
                 df_lr=1e-3,
                 **kwargs):
        super(DiscreteSkillTrainerSeqwiseStepwise, self).__init__(
            *args,
            skill_prior_dist=skill_prior_dist,
            loss_fun=loss_fun,
            optimizer_class=optimizer_class,
            **kwargs
        )

        self.df_train_counter = 0

    def create_optimizer_step(self, optimizer_class, df_lr):
        return optimizer_class(
            chain(
                self.df.classifier_step.parameters(),
                self.df.pos_encoder.parameters(),
            ),
            lr=df_lr
        )

    def _df_loss_step_rewards(
            self,
            loss_calc_values: dict,
            skills: torch.Tensor,
    ):
        """"
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

        skills = skills.view(batch_size * seq_len, skill_dim)

        hidden_feature_seq = loss_calc_values['hidden_feature_seq']
        recon_feature_seq = loss_calc_values['recon_feature_seq']['dist']
        post_skills = loss_calc_values['post_skills']['dist']

        assert hidden_feature_seq.shape == torch.Size(
            (batch_size,
             seq_len,
             self.df.rnn_params['num_features_hs_posenc']))
        hidden_feature_seq = hidden_feature_seq.reshape(batch_size * seq_len, -1)

        assert post_skills.batch_shape == skills.shape
        assert hidden_feature_seq.shape == recon_feature_seq.batch_shape

        # Rewards
        rewards = post_skills.log_prob(skills)
        rewards = torch.sum(rewards, dim=data_dim, keepdim=True)
        rewards = rewards.reshape(batch_size, seq_len, 1)

        # Reshape Dist
        pri_dist = self.skill_prior(hidden_feature_seq)
        assert len(pri_dist.batch_shape) == 2
        pri = dict(
            dist=pri_dist,
            sample=pri_dist.sample()
        )

        # Reshape Dist
        post_dist = post_skills
        post = dict(
            dist=post_dist,
            sample=post_dist.rsample()
        )

        # Reshape Dist
        recon_feature_seq_dist = recon_feature_seq
        assert len(recon_feature_seq_dist.batch_shape) == 2
        recon = dict(
            dist=recon_feature_seq_dist,
            sample=recon_feature_seq_dist.loc,
        )

        # Loss Calculation
        info_loss, log_dict = self.loss_fun(
            pri=pri,
            post=post,
            recon=recon,
            data=hidden_feature_seq.detach(),
            latent_guide=skills,
        )

        return dict(
            df_loss=info_loss,
            rewards=rewards,
            log_dict=log_dict
        )

    def train_whole_df_only(self, batch):
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

        self.df_optimizer_seq.zero_grad()
        df_loss['seq'].backward()
        self.df_optimizer_seq.step()

        self.df_optimizer_step.zero_grad()
        df_loss['step'].backward()
        self.df_optimizer_step.step()

    def train_df_step_only(self, batch):
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

        self.df_optimizer_step.zero_grad()
        df_loss_step.backward()
        self.df_optimizer_step.step()

    def train_from_torch(self, batch):
        if self.df_train_counter % 5 == 0:
            super(DiscreteSkillTrainerSeqwiseStepwise, self).train_from_torch(batch)

        else:
            self.train_df_step_only(batch)

        self.df_train_counter += 1
