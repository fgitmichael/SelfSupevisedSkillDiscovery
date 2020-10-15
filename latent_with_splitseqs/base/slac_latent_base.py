import abc
import torch

from latent_with_splitseqs.base.latent_base \
    import StochasticLatentNetBase


class SlacLatentBase(StochasticLatentNetBase):

    def __init__(self,
                 *args,
                 latent1_dim,
                 latent2_dim,
                 **kwargs):
        super(SlacLatentBase, self).__init__(*args, **kwargs)

        self.latent1_dim = latent1_dim
        self.latent2_dim = latent2_dim

    @property
    def latent_dim(self):
        return self.latent1_dim + self.latent2_dim

    def sample_posterior_samples_cat(self,
                                    *args,
                                    **kwargs):
        data_dim = -1

        post_dict = self.sample_posterior(*args, **kwargs)
        latent1_samples = post_dict.pop('latent1_samples')
        latent2_samples = post_dict.pop('latent2_samples')
        latent_samples = torch.cat([latent1_samples,
                                    latent2_samples], dim=data_dim)

        return dict(
            latent_dists=post_dict['latent1_dists'],
            latent_samples=latent_samples,
        )

    def sample_prior_samples_cat(self, *args, **kwargs):
        data_dim = -1

        pri_dict = self.sample_prior(*args, **kwargs)
        latent1_samples = pri_dict.pop('latent1_samples')
        latent2_samples = pri_dict.pop('latent2_samples')
        latent_samples = torch.cat([latent1_samples,
                                    latent2_samples], dim=data_dim)

        return dict(
            latent_dists=pri_dict['latent1_dists'],
            latent_samples=latent_samples,
        )
