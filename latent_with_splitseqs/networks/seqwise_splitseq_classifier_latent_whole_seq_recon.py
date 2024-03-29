import torch

from diayn_seq_code_revised.networks.my_gaussian import MyGaussian as Gaussian

from latent_with_splitseqs.latent.slac_latent_conditioned_on_skill_seq \
    import SlacLatentNetConditionedOnSkillSeq
from latent_with_splitseqs.base.classifier_base import SplitSeqClassifierBase


class SeqwiseSplitseqClassifierSlacLatentWholeSeqRecon(SplitSeqClassifierBase):

    def __init__(self,
                 *args,
                 seq_len: int,
                 obs_dim: int,
                 skill_dim: int,
                 latent_net: SlacLatentNetConditionedOnSkillSeq,
                 hidden_units_classifier=(256, 256),
                 leaky_slope_classifier=0.2,
                 classifier_dropout=0.3,
                 classifier_std=None,
                 **kwargs
                 ):
        super(SeqwiseSplitseqClassifierSlacLatentWholeSeqRecon, self).__init__(
            *args,
            obs_dim=obs_dim,
            seq_len=seq_len,
            **kwargs,
        )

        self.latent_net = latent_net
        self.classifier = Gaussian(
            input_dim=self.latent_net.latent1_dim +
                      self.latent_net.latent2_dim,
            output_dim=skill_dim,
            hidden_units=hidden_units_classifier,
            leaky_slope=leaky_slope_classifier,
            dropout=classifier_dropout,
            std=classifier_std,
        )

        self.obs_dim = obs_dim
        self.skill_dim = skill_dim

    @torch.no_grad()
    def eval_forwardpass(self,
                         obs_seq,
                         skill=None):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = obs_seq.size(batch_dim)
        seq_len = obs_seq.size(seq_dim)

        if skill is None:
            latent = self.latent_net.sample_prior(
                obs_seq=obs_seq
            )

        else:
            latent = self.latent_net.sample_posterior(
                skill=skill,
                obs_seq=obs_seq,
            )

        latent_seq = torch.cat(
            [latent['latent1_samples'],
             latent['latent2_samples']],
            dim=data_dim)
        skill_recon_dist = self.classifier(latent_seq[:, -1, :])
        assert skill_recon_dist.batch_shape[batch_dim] == batch_size
        assert len(skill_recon_dist.batch_shape) == 2

        return dict(
            skill_recon_dist=skill_recon_dist,
            feature_seq=latent_seq,
        )

    def train_forwardpass(self,
                          obs_seq,
                          skill,):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = obs_seq.size(batch_dim)
        seq_len = obs_seq.size(seq_dim)

        pri_post_dict = self.latent_net(
            skill=skill,
            obs_seq=obs_seq,
        )

        pri = pri_post_dict['pri']
        post = pri_post_dict['post']
        latent_seq = torch.cat(
            [post['latent1_samples'],
             post['latent2_samples']],
            dim=data_dim
        )

        skill_recon_dist = self.classifier(
            latent_seq[:, 1:, :].reshape(
                batch_size * seq_len,
                latent_seq.size(data_dim)
            )
        )
        assert skill_recon_dist.batch_shape[:data_dim] \
               == torch.Size((batch_size * seq_len,))
        assert skill_recon_dist.batch_shape[data_dim] == skill.shape[data_dim]

        return dict(
            latent_pri=pri,
            latent_post=post,
            recon=skill_recon_dist,
        )
