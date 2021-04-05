import torch
import torch.nn.functional as F
import numpy as np

from diayn_with_rnn_classifier.trainer.diayn_trainer_modularized import \
    DIAYNTrainerModularized

from code_slac.network.latent import Gaussian

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict


class DIAYNContTrainer(DIAYNTrainerModularized):

    def __init__(self,
                 *args,
                 df: Gaussian,
                 **kwargs):
        assert isinstance(df, Gaussian)
        super().__init__(*args, df=df, **kwargs)

    def _df_loss_intrinsic_reward(self,
                                  skills,
                                  next_obs):
        assert isinstance(self.df, Gaussian)

        d_pred_dist = self.df(next_obs)
        log_prob_skills = d_pred_dist.log_prob(skills)

        rewards = torch.sum(log_prob_skills, dim=-1, keepdim=True)
        df_loss = - torch.sum(log_prob_skills, dim=-1).mean()
        pred_skill = d_pred_dist.loc

        return dict(
            df_loss=df_loss,
            rewards=rewards,
            pred_z=pred_skill,
            z_hat=skills,
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
        df_accuracy = F.mse_loss(z_hat, pred_z, reduction='sum')

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

    def get_snapshot(self) -> dict:
        return super(DIAYNTrainerModularized, self).get_snapshot()
