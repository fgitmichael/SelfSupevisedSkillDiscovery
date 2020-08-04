import torch
import math
from torch.nn import functional as F

from diayn_with_rnn_classifier.trainer.seq_wise_trainer_with_diayn_classifier_vote import \
    DIAYNTrainerMajorityVoteSeqClassifier
from diayn_with_rnn_classifier.trainer.diayn_trainer_modularized import \
    DIAYNTrainerModularized
from diayn_rnn_seq_rnn_stepwise_classifier.networks.bi_rnn_stepwise import \
    BiRnnStepwiseClassifier

import self_supervised.utils.my_pytorch_util as my_ptu


class DIAYNStepWiseRnnTrainer(DIAYNTrainerMajorityVoteSeqClassifier):

    def _df_loss_intrinsic_reward(self,
                                  skills,
                                  next_obs):
        """
        Args:
            skills            : (N, S, skill_dim)
            next_obs          : (N, S, obs_dim)
        Return:
            df_loss           : (N, 1)
            rewards           : (N, S, 1)
            pred_z            : (N, S) skill predicted skill ID's
            z_hat             : (N, S) skill ground truth skill ID's
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = next_obs.size(batch_dim)
        seq_len = next_obs.size(seq_dim)
        obs_dim = next_obs.size(data_dim)
        skill_dim = skills.size(data_dim)

        z_hat = torch.argmax(skills, dim=data_dim)
        d_pred = self.df(next_obs)

        d_pred_log_softmax = F.log_softmax(d_pred, data_dim)
        pred_z = torch.argmax(d_pred_log_softmax, dim=data_dim)

        b, s = torch.meshgrid(
            [torch.arange(batch_size),
             torch.arange(seq_len)]
        )
        assert b.shape == s.shape == z_hat.shape == torch.Size((batch_size, seq_len))
        rewards = d_pred_log_softmax[b, s, z_hat] - math.log(1/self.policy.skill_dim)
        assert rewards.shape == b.shape
        rewards = rewards.unsqueeze(dim=data_dim)

        assert my_ptu.tensor_equality(
            d_pred.view(batch_size * seq_len, self.num_skills)[:seq_len],
            d_pred[0]
        )
        assert my_ptu.tensor_equality(
            z_hat.view(batch_size * seq_len)[:seq_len],
            z_hat[0].squeeze()
        )
        df_loss = self.df_criterion(
            d_pred.view(batch_size * seq_len, self.num_skills),
            z_hat.view(batch_size * seq_len)
        )

        assert z_hat.shape == pred_z.shape == torch.Size((batch_size, seq_len,))
        assert rewards.shape == torch.Size((batch_size, seq_len, 1))

        return dict(
            df_loss=df_loss,
            rewards=rewards,
            pred_z=pred_z,
            z_hat=z_hat
        )
