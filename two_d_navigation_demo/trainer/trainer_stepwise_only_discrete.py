import torch
import numpy as np
import math
from torch.nn import functional as F

from operator import itemgetter

from diayn_seq_code_revised.trainer.trainer_seqwise_stepwise_revised import \
    DIAYNAlgoStepwiseSeqwiseRevisedTrainer

import self_supervised.utils.my_pytorch_util as my_ptu

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict


class StepwiseOnlyDiscreteTrainer(DIAYNAlgoStepwiseSeqwiseRevisedTrainer):

    def create_optimizer_seq(self, optimizer_class, df_lr):
        return None

    def create_optimizer_step(self, optimizer_class, df_lr):
        return optimizer_class(
            self.df.parameters(),
            lr=df_lr
        )

    def _df_loss_intrinsic_reward(self,
                                  skills_and_id_dict,
                                  next_obs):
        df_ret_dict = self.df(
            next_obs,
            train=True
        )

        classified_steps, \
        hidden_features = itemgetter(
            'classified_steps',
            'hidden_features_seq'
        )(df_ret_dict)

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
            df_loss=df_loss_step,
            rewards=rewards,
            pred_z=pred_z_step,
            z_hat=z_hat_step
        )

    def _update_networks(self,
                         df_loss,
                         qf1_loss,
                         qf2_loss,
                         policy_loss):
        self.df_optimizer_step.zero_grad()
        df_loss.backward()
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

    def _calc_df_accuracy(self, z_hat, pred_z):
        """
        Args:
            z_hat               : (N, S,) skill_gt_id stepwise
            pred_z              : (N, S,) prediction stepwise
        Return:
            step                      : scalar
        """
        assert z_hat.shape == pred_z.shape

        df_accuracy = torch.sum(
            torch.eq(
                z_hat,
                pred_z
            )
        ).float()/pred_z.view(-1, 1).size(0)

        return df_accuracy

    def _df_loss_seq(self,
                     d_pred_seq,
                     skills_and_id_dict):
        raise NotImplementedError('Step wise only!')

    @property
    def num_skills(self):
        return self.df.num_skills

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
        df_accuracy = self._calc_df_accuracy(
            z_hat=z_hat,
            pred_z=pred_z
        )

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['Intrinsic Rewards'] = \
                np.mean(ptu.get_numpy(rewards))
            self.eval_statistics['DF Loss'] = \
                np.mean(ptu.get_numpy(df_loss))
            self.eval_statistics['DF Accuracy'] = \
                np.mean(ptu.get_numpy(df_accuracy))
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

#    def _df_loss_step_rewards(
#            self,
#            d_pred_step,
#            skills_and_id_dict,
#    ):
#        """
#        Args:
#            d_pred_step         : (N, S, num_skills)
#            skills_and_id_dict
#                skills          : (N, S, skill_dim)
#                ids             : (N, S, 1)
#        Return:
#            df_loss_step        : scalar tensor
#            rewards             : (N, S, 1)
#            z_hat_step          : (N, S)
#            pred_z_step         : (N, S)
#        """
#        skills = skills_and_id_dict['skills']
#        skills_id = skills_and_id_dict['ids']
#
#        batch_dim = 0
#        seq_dim = 1
#        data_dim = -1
#        batch_size = skills.size(batch_dim)
#        seq_len = skills.size(seq_dim)
#        skill_dim = skills.size(data_dim)
#
#        assert d_pred_step.shape == torch.Size(
#            (batch_size,
#             seq_len,
#             self.num_skills)
#        )
#
#        assert skills_id.shape == torch.Size((batch_size, seq_len, 1))
#        z_hat_step = skills_id[..., 0]
#        assert z_hat_step.shape == torch.Size(
#            (batch_size,
#             seq_len))
#
#        d_pred_logsoftmax = F.log_softmax(d_pred_step, dim=data_dim)
#        pred_z_step = torch.argmax(d_pred_logsoftmax, dim=data_dim)
#
#        # Rewards
#        b, s = torch.meshgrid(
#            [torch.arange(batch_size),
#             torch.arange(seq_len)]
#        )
#        assert b.shape \
#               == s.shape \
#               == z_hat_step.shape \
#               == torch.Size((batch_size, seq_len))
#        rewards = d_pred_logsoftmax[b, s, z_hat_step] - math.log(1/self.policy.skill_dim)
#        rewards = rewards.unsqueeze(dim=data_dim)
#        assert rewards.shape == torch.Size((batch_size, seq_len, 1))
#
#        # Loss
#        assert my_ptu.tensor_equality(
#            d_pred_step.view(
#                batch_size * seq_len,
#                self.num_skills
#            )[:seq_len],
#            d_pred_step[0]
#        )
#        assert my_ptu.tensor_equality(
#            z_hat_step.view(batch_size * seq_len)[:seq_len],
#            z_hat_step[0]
#        )
#        df_step_loss = self.df_criterion(
#            d_pred_step.view(
#                batch_size * seq_len,
#                self.num_skills
#            ),
#            z_hat_step.view(batch_size * seq_len)
#        )
#






