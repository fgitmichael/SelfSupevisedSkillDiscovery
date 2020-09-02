import torch
from operator import itemgetter

from seqwise_cont_skillspace.trainer.cont_skillspace_seqwise_trainer import \
    ContSkillTrainerSeqwiseStepwise


class ContSkillTrainerSeqwiseStepwiseStepRepeatTrainer(ContSkillTrainerSeqwiseStepwise):

    def __init__(self,
                 *args,
                 step_training_repeat=1,
                 **kwargs):
        super(ContSkillTrainerSeqwiseStepwiseStepRepeatTrainer, self).__init__(
            *args,
            **kwargs
        )
        self.step_repeat = step_training_repeat
        self.train_cnt = 0

    def _train_from_torch_step_only(self, batch):
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

        """
        Update networks
        """
        self._update_networks_step_only(
            df_loss=df_loss,
        )

    def _update_networks_step_only(self, df_loss):
        self.df_optimizer_step.zero_grad()
        df_loss['step'].backward()
        self.df_optimizer_step.step()

    def _df_loss_intrinsic_reward_step_only(self,
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
                step=df_loss_step,
            ),
            rewards=rewards,
        )

    def train_from_torch(self, batch):
        if self.train_cnt % self.step_repeat == 0:
            super().train_from_torch(batch)
        else:
            self._train_from_torch_step_only(batch)
