from latent_with_splitseqs.trainer.rnn_with_splitseqs_trainer_end_recon_only import \
    URLTrainerRnnWithSplitseqsEndReconOnly

from code_slac.utils import update_params


class URLTrainerRnnWithSplitseqsWholeSeqRecon(
    URLTrainerRnnWithSplitseqsEndReconOnly):

    def train_latent_from_torch(self, batch):
        self._check_latent_batch(batch)

        next_obs = batch['next_obs']
        skills = batch['mode']
        batch_size, seq_len, skill_dim = skills.shape
        skills_reshaped = skills.reshape(
            batch_size * seq_len,
            skill_dim
        )

        # Skill is not needed as posterior coincides with prior
        assert self.df.training is True
        df_ret_dict = self.df(
            obs_seq=next_obs,
            skill=None,
        )
        skill_recon_dist = df_ret_dict['skill_recon_dist']

        pri_dist = self.skill_prior_dist(skills_reshaped)
        df_loss, log_dict = self.loss_fun(
            pri=dict(
                dist=pri_dist,
                sample=pri_dist.sample(),
            ),
            post=dict(
                dist=skill_recon_dist,
                sample=skill_recon_dist.rsample()
            ),
            recon=None,
            guide=skills_reshaped,
            data=None,
        )

        update_params(
            optim=self.df_optimizer,
            network=self.df,
            loss=df_loss,
        )

        if self._need_to_update_eval_statistics:
            self.eval_statistics['latent/df_loss'] = df_loss.item()
            for k, v in log_dict.items():
                self.eval_statistics['latent/' + k] = v.item()
