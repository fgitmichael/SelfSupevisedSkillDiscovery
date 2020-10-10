from latent_with_splitseqs.trainer.latent_with_splitseqs_trainer \
    import URLTrainerLatentWithSplitseqs


class URLTrainerLatentWithSplitseqsRnnEndReconOnly(URLTrainerLatentWithSplitseqs):

    def train_latent_from_torch(self, batch):
        self._check_latent_batch(batch)
        batch_dim = 0

        next_obs = batch['next_obs']
        skills = batch['mode']

        skill = skills[:, 0, :]
        df_ret_dict = self.df(
            obs_seq=next_obs,
            skill=skill
        )
        skill_recon_dist = df_ret_dict['skill_recon_dist']

        #df_loss = -skill_recon_dist.log_prob(skill).mean(dim=batch_dim).sum()
        pri_dist = self.skill_prior_dist(skill)
        df_loss, log_dict = self.loss_fun(
            pri=dict(
                dist=pri_dist,
                sample=pri_dist.sample(),
            ),
            post=dict(
                dist=skill_recon_dist,
                sample=skill_recon_dist.rsample(),
            ),
            recon=None,
            guide=skill,
            data=None,
        )

        # Update network
        self.df_optimizer.zero_grad()
        df_loss.backward()
        self.df_optimizer.step()

        if self._need_to_update_eval_statistics:
            self.eval_statistics['latent/df_loss'] = df_loss.item()
            for k, v in log_dict.items():
                self.eval_statistics['latent/' + k] = v.item()

    def _latent_loss(self,
                     skills,
                     next_obs):
        raise NotImplementedError
