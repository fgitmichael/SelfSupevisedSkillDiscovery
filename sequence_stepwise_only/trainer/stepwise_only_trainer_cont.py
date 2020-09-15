import torch
import numpy as np
from operator import itemgetter

from seqwise_cont_skillspace.trainer.cont_skillspace_seqwise_trainer import \
    ContSkillTrainerSeqwiseStepwise

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict


class StepwiseOnlyTrainerCont(ContSkillTrainerSeqwiseStepwise):

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

    def create_optimizer_seq(self, optimizer_class, df_lr):
        return None

    def create_optimizer_step(self, optimizer_class, df_lr):
        return optimizer_class(
            self.df.parameters(),
            lr=df_lr
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

        classified_steps, \
        feature_recon_dist, \
        hidden_features_seq = itemgetter(
            'classified_steps',
            'feature_recon_dist',
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
            log_dict=log_dict_df_step
        )

    def _df_loss_seq(self,
                     pred_skills_seq,
                     skills):
        raise NotImplementedError('Stepwise Only!')

    def _df_loss_step_rewards(
            self,
            loss_calc_values: dict,
            skills: torch.Tensor,
    ):
        """
        Args:
            loss_calc_values:
                hidden_features_seq         : (N, S, hidden_size_rnn * num_direction)
                recon_features_seq          : (N, S, hidden_size_rnn * num_direction)
                                              dist and samples
                post_skills                 : (N, S, skill_dim) - latentspace vectors
                                              dist and samples
            skills                          : (N, S, skill_dim)
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
        recon_feature_seq = loss_calc_values['recon_feature_seq']['dist']
        post_skills = loss_calc_values['post_skills']['dist']

        assert hidden_feature_seq.shape == torch.Size(
            (batch_size,
             seq_len,
             self.df.rnn_params['num_features_hs_posenc']))
        assert post_skills.batch_shape == skills.shape
        assert hidden_feature_seq.shape == recon_feature_seq.batch_shape

        rewards = post_skills.log_prob(post_skills.loc)
        rewards = torch.sum(rewards, dim=data_dim, keepdim=True)

        # Reshape Dist
        pri_dist = self.reshape_dist(self.skill_prior(hidden_feature_seq))
        assert len(pri_dist.batch_shape) == 2
        pri = dict(
            dist=pri_dist,
            sample=pri_dist.sample()
        )

        # Reshape Dist
        post_dist = self.reshape_dist(post_skills)
        post = dict(
            dist=post_dist,
            sample=post_dist.rsample()
        )

        # Reshape Dist
        recon_feature_seq_dist = self.reshape_dist(recon_feature_seq)
        assert len(recon_feature_seq_dist.batch_shape) == 2
        recon = dict(
            dist=recon_feature_seq_dist,
            sample=recon_feature_seq_dist.loc,
        )

        # Loss Calculation
        hidden_feature_seq_data_dim = hidden_feature_seq.size(data_dim)
        hidden_feature_seq = hidden_feature_seq.reshape(
            batch_size * seq_len,
            hidden_feature_seq_data_dim,
            )
        skills = skills.reshape(
            batch_size * seq_len,
            skill_dim
        )
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
            log_dict=log_dict,
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
            #self.eval_statistics.update(create_stats_ordered_dict(
            #    'D Predictions Step',
            #    ptu.get_numpy(pred_z['step']),
            #))
            #self.eval_statistics.update(create_stats_ordered_dict(
            #    'D Predictions Seq',
            #    ptu.get_numpy(pred_z['seq']),
            #))
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

    def _update_networks(self,
                         df_loss,
                         qf1_loss,
                         qf2_loss,
                         policy_loss):
        self.df_optimizer_step.zero_grad()
        df_loss['step'].backward()
        self.df_optimizer_step.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
