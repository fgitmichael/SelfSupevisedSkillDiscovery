import torch
import numpy as np
import torch.distributions as torch_dist
from operator import itemgetter

from latent_with_splitseqs.trainer.latent_with_splitseqs_trainer \
    import URLTrainerLatentWithSplitseqs

from code_slac.utils import update_params,calc_kl_divergence


class URLTrainerLatentSplitSeqsSRNNEndReconOnly(URLTrainerLatentWithSplitseqs):

    def _check_latent_outputs(
            self,
            latent_pri: dict,
            latent_post: dict,
            skill: torch.Tensor,
            recon: torch_dist.Distribution,
            seq_len: int,
            batch_size: int,
            skill_dim: int,
    ):
        assert isinstance(recon, torch_dist.Distribution)
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        assert latent_pri['latent_samples'].shape == latent_post['latent_samples'].shape
        assert len(latent_post['latent_dists']) \
               == len(latent_pri['latent_dists']) \
               == latent_pri['latent_samples'].size(seq_dim) \
               == seq_len
        assert latent_pri['latent_dists'][0].batch_shape[batch_dim] \
               == latent_post['latent_dists'][0].batch_shape[batch_dim] \
               == latent_post['latent_samples'].size(batch_dim) \
               == skill.size(batch_dim) \
               == batch_size
        assert latent_post['latent_dists'][0].batch_shape \
               == latent_pri['latent_dists'][0].batch_shape

    def _latent_loss(self,
                     skills,
                     next_obs):
        """
        Args:
            skills                      : (N, S, skill_dim) tensor
            next_obs                    : (N, S, obs_dim) tensor
        Returns:
            df_loss                     : scalar tensor
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1

        seq_len = next_obs.size(seq_dim)
        batch_size = next_obs.size(batch_dim)
        obs_dim = next_obs.size(data_dim)
        skill = skills[:, 0, :]
        skill_dim = skill.size(data_dim)

        df_ret_dict = self.df(
            obs_seq=next_obs,
            skill=skill,
        )
        latent_pri, \
        latent_post, \
        recon = itemgetter(
            'latent_pri',
            'latent_post',
            'recon',
        )(df_ret_dict)

        self._check_latent_outputs(
            latent_pri=latent_pri,
            latent_post=latent_post,
            skill=skill,
            recon=recon,
            seq_len=seq_len,
            batch_size=batch_size,
            skill_dim=skill_dim,
        )

        kld_loss = calc_kl_divergence(
            latent_post['latent_dists'],
            latent_pri['latent_dists']
        )

        skill_prior_dist = self.skill_prior_dist(recon.sample())
        recon_loss, log_dict = self.loss_fun(
            pri=dict(
                dist=skill_prior_dist,
                sample=skill_prior_dist.sample()
            ),
            post=dict(
                dist=recon,
                sample=recon.rsample(),
            ),
            recon=None,
            guide=skill,
            data=None,
        )

        if self.df.latent_net.beta is not None:
            self.df.latent_net.anneal_beta()
            beta = self.df.latent_net.beta

        else:
            beta = 1.
        latent_loss = beta * kld_loss + recon_loss

        return latent_loss, dict(
            kld_loss=kld_loss.item(),
            recon_info_loss=recon_loss.item(),
            beta=beta,
            **{'recon' + k: el.item() for k, el in log_dict.items()}
        )
