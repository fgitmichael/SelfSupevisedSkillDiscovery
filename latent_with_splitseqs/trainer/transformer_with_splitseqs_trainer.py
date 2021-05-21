from latent_with_splitseqs.trainer.latent_with_splitseqs_trainer \
    import URLTrainerLatentWithSplitseqs
from latent_with_splitseqs.networks.seqwise_splitseq_classifier_transformer \
    import SeqwiseSplitseqClassifierTransformer

from code_slac.utils import update_params


class URLTrainerTransformerWithSplitseqs(URLTrainerLatentWithSplitseqs):

    def __init__(self,
                 *args,
                 df,
                 **kwargs):
        assert isinstance(df, SeqwiseSplitseqClassifierTransformer)
        super().__init__(
            *args,
            df=df,
            **kwargs
        )

    def train_latent_from_torch(self, batch):
        self._check_latent_batch(batch)

        next_obs = batch['next_obs']
        skills = batch['mode']

        skill = skills[:, 0, :]
        assert self.df.training is True
        df_ret_dict = self.df(
            obs_seq=next_obs,
            skill=skill,
        )
        skill_recon_dist = df_ret_dict['skill_recon_dist']

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

        update_params(
            optim=self.df_optimizer,
            network=self.df,
            loss=df_loss,
        )

        if self._need_to_update_eval_statistics:
            self.eval_statistics['latent/df_loss'] = df_loss.item()
            for k, v in log_dict.items():
                self.eval_statistics['latent/' + k] = v.item()

    def _latent_loss(self,
                     skills,
                     next_obs):
        raise NotImplementedError
