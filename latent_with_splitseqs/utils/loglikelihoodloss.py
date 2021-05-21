import torch

from seqwise_cont_skillspace.utils.info_loss import GuidedInfoLoss

from code_slac.utils import calc_kl_divergence

from diayn_seq_code_revised.networks.my_gaussian import ConstantGaussianMultiDimMeanSpec

import rlkit.torch.pytorch_util as ptu


class GuidedKldLogOnlyLoss(GuidedInfoLoss):
    """
    Only LogLikelyhood-Loss (compared to base class)
    """
    def loss(self,
             pri: dict,
             post: dict,
             recon: dict,
             data: torch.Tensor,
             dist_key=None,
             sample_key=None,
             **kwargs):
        if not 'guide' in kwargs:
            raise ValueError("Guide has to be passed!")
        if len(list(kwargs.keys())) > 1:
            raise ValueError("Values that are not used are passed!")

        guide = kwargs['guide']

        if dist_key is not None:
            self.dist_key = dist_key

        if sample_key is not None:
            self.sample_key = sample_key

        self.input_assertions(
            post=post,
            pri=pri,
            recon=recon,
            data=data
        )

        data_loss_dict = self._data_loss(
            dict(recon=recon,
                 data=data,
                 post=post,
                 guide=guide
                 )
        )

        latent_loss_dict = self._latent_loss(
            post=post,
            pri=pri,
        )

        loss = data_loss_dict['loss'] + latent_loss_dict['kld_weighted']
        log_dict = dict(
            **latent_loss_dict['log_dict'],
            **data_loss_dict['log_dict'],
        )

        return loss, log_dict

    def _latent_loss(self, post, pri):
        # Don't use given prior, use gaussian with specified mean instead
        pri = None
        data_dim = -1
        pri_dist_creator = ConstantGaussianMultiDimMeanSpec(
            output_dim=post[self.dist_key].batch_shape[data_dim]
        )
        kld = calc_kl_divergence([post[self.dist_key]],
                                 [pri_dist_creator(mean=post[self.dist_key].loc)])
        kld_weighted = (1 - self.alpha) * kld

        log = dict(
            kld=kld,
            kld_weighted=kld_weighted,
        )

        return dict(
            kld_weighted=kld_weighted,
            log_dict=log,
        )


class GuidedKldLogOnlyLossBounded(GuidedKldLogOnlyLoss):

    def _latent_loss(self, post, pri):
        # Don't use given prior, use gaussian with specified mean instead
        pri = None
        data_dim = -1
        pri_dist_creator = ConstantGaussianMultiDimMeanSpec(
            output_dim=post[self.dist_key].batch_shape[data_dim]
        )
        pri_dist = pri_dist_creator(mean=post[self.dist_key].loc)
        kld_real = calc_kl_divergence([post[self.dist_key]],
                                      [pri_dist_creator(mean=post[self.dist_key].loc)])
        pri_dist_std = torch.sum(pri_dist.scale, dim=data_dim, )
        post_dist_std = torch.sum(post[self.dist_key].scale, dim=data_dim, )
        idx = pri_dist_std > post_dist_std
        post[self.dist_key].loc, \
        post[self.dist_key].scale = post[self.dist_key].loc[idx, :], \
                                    post[self.dist_key].scale[idx, :]
        kld_bound = calc_kl_divergence([post[self.dist_key]],
                                       [pri_dist_creator(mean=post[self.dist_key].loc)])
        if torch.isnan(kld_bound):
            kld_weighted = ptu.zeros_like(kld_real)
        else:
            kld_weighted = (1 - self.alpha) * kld_bound

        log = dict(
            kld_bound=kld_bound,
            kld=kld_real,
            kld_weighted=kld_weighted,
        )

        return dict(
            kld_weighted=kld_weighted,
            log_dict=log,
        )
