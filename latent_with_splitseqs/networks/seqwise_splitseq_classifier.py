import torch
import torch.distributions as torch_dist
from operator import itemgetter
from code_slac.network.base import BaseNetwork
from code_slac.network.latent import Gaussian

from latent_with_splitseqs.networks.slac_latent_net \
    import SlacLatentNetConditionedOnSingleSkill

class SeqwiseSplitseqClassifierSlacLatent(BaseNetwork):

    def __init__(self,
                 seq_len: int,
                 obs_dim: int,
                 skill_dim: int,
                 latent_net: SlacLatentNetConditionedOnSingleSkill,
                 hidden_units_classifier=(256, 256),
                 leaky_slope_classifier=0.2,
                 ):
        super(SeqwiseSplitseqClassifierSlacLatent, self).__init__()

        self.latent_net = latent_net
        self.classifier = Gaussian(
            input_dim=self.latent_net.latent1_dim +
                      self.latent_net.latent2_dim,
            output_dim=skill_dim,
            hidden_units=hidden_units_classifier,
            leaky_slope=leaky_slope_classifier,
        )

        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.skill_dim = skill_dim

    def forward(self,
                obs_seq,
                skill=None):
        """
        Args:
            skill                   : (N, skill_dim) tensor (skill batch)
            obs_seq                 : (N, S, obs_dim) tensor (sequence batch)
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        if skill is not None:
            assert skill.size(batch_dim) == obs_seq.size(batch_dim)
            assert skill.size(data_dim) == self.skill_dim
            assert len(skill.shape) == 2
        assert obs_seq.size(seq_dim) == self.seq_len
        assert len(obs_seq.shape) == 3

        if self.training:
            return self.train_forwardpass(
                obs_seq=obs_seq,
                skill=skill,
            )

        else:
            return self.eval_forwardpass(
                obs_seq=obs_seq,
                skill=skill,
            )

    @torch.no_grad()
    def eval_forwardpass(self,
                         obs_seq,
                         skill=None):
        data_dim = -1

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

        return dict(
            skill_recon_dist=skill_recon_dist,
            feature_seq=latent_seq,
        )

    def train_forwardpass(self,
                          obs_seq,
                          skill,):
        pri_post_dict = self.latent_net(
            skill=skill,
            obs_seq=obs_seq,
        )
        pri, post = itemgetter('pri',
                               'post')(pri_post_dict)

        post_latent1_samples, post_latent2_samples = itemgetter(
            'latent1_samples',
            'latent2_samples'
        )(post)
        skill_recon_dist = self.classifier(
            [post_latent1_samples[:, -1, :],
             post_latent2_samples[:, -1, :]]
        )

        return dict(
            latent_pri=pri,
            latent_post=post,
            recon=skill_recon_dist,
        )