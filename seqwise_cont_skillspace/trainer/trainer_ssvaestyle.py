import torch
import torch.distributions as torch_dist
from torch import nn
from itertools import chain
from operator import itemgetter

from seqwise_cont_skillspace.trainer.cont_skillspace_seqwise_trainer import \
    ContSkillTrainerSeqwiseStepwise
from seqwise_cont_skillspace.trainer.discrete_skillspace_seqwise_stepwise_revised_trainer \
    import DiscreteSkillTrainerSeqwiseStepwise


class SsvaestyleSkillTrainer(DiscreteSkillTrainerSeqwiseStepwise):

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
        info_loss, log_dict = self.loss_fun(
            pri=pri,
            post=post,
            recon=recon,
            data=hidden_feature_seq.detach()
        )

        return dict(
            df_loss=info_loss,
            rewards=rewards,
            log_dict=log_dict,
        )
