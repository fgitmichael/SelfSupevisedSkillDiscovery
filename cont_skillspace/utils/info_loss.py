import torch
from torch.nn import functional as F

from mode_disent.utils.mmd import compute_mmd_tutorial
from code_slac.utils import calc_kl_divergence


class InfoLoss:
    def __init__(self,
                 alpha: float,
                 lamda: float):
        self.alpha = alpha
        self.lamda = lamda

        self.dist_key = 'dist'
        self.sample_key = 'sample'

    def loss(self,
             pri: dict,
             post: dict,
             recon: dict,
             data: torch.Tensor,
             dist_key=None,
             sample_key=None):
        """
        Args:
            pri
                dist                : (N, latent_dim) distribution
                sample              : (N, latent_dim) samples
            post
                dist                : (N, latent_dim) distribution
                sample              : (N, latent_dim) samples
            recon
                dist                : (N, data_dim) distribution
                sample              : (N, data_dim) samples
                                      (for gaussians typically loc instead of samples)
            data                    : (N, ..., data_dim) tensor of original data

        Return:
            loss                    : scalar tensor
            log_values
                kld                 : scalar tensor
                mmd                 : scalar tensor
                mse                 : scalar tensor
                ll                  : scalar tensor
                kld_info            : scalar tensor
                mmd_info            : scalar tensor
                loss_latent         : scalar tensor
                loss_data           : scalar tensor
                info_loss           : scalar tensor
        """
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

        latent_loss_dict = self._latent_loss(
            post=post,
            pri=pri
        )
        data_loss_dict = self._data_loss(
            dict(recon=recon,
                 data=data,
                 post=post,)
        )

        info_loss = data_loss_dict['loss'] + \
                    latent_loss_dict['kld_info'] + \
                    latent_loss_dict['mmd_info']

        log_dict = {**latent_loss_dict['log_dict'],
                    **data_loss_dict['log_dict']}

        return info_loss, log_dict

    def _latent_loss(self, post, pri):
        kld = calc_kl_divergence([post[self.dist_key]],
                                 [pri[self.dist_key]])

        mmd = compute_mmd_tutorial(pri[self.sample_key],
                                   post[self.sample_key])

        kld_info = (1 - self.alpha) * kld
        mmd_info = (self.alpha + self.lamda - 1) * mmd

        log = dict(
            kld=kld,
            mmd=mmd,
            kld_info=kld_info,
            mmd_info=mmd_info,
            loss_latent=kld_info + mmd_info,
        )

        return dict(
            kld_info=kld_info,
            mmd_info=mmd_info,
            log_dict=log,
        )

    def _data_loss(self, data_dict):
        """
        Args:
            data_dict
                post                : (N, latent_dim) dist and samples
                recon               : (N, data_dim) dist and samples
                data                : (N, data_dim) tensor
        Return:
            loss                    : scalar tensor
            log_dict                : dictionary
        """
        mse = F.mse_loss(
            data_dict['recon'][self.dist_key].loc,
            data_dict['data'])

        log = dict(
            mse=mse,
            loss_data=mse,
        )

        return dict(
            loss=mse,
            log_dict=log,
        )

    def input_assertions(self,
                         post,
                         pri,
                         recon,
                         data):
        dist_key = self.dist_key
        sample_key = self.sample_key

        if not data.is_contiguous():
            data = data.contiguous()
        data = data.view(-1, data.size(-1))
        assert data.shape == recon[dist_key].batch_shape

        batch_dim = 0
        assert pri[dist_key].batch_shape \
               == pri[sample_key].shape \
               == post[dist_key].batch_shape \
               == post[sample_key].shape
        assert len(pri[dist_key].batch_shape) == 2
        batch_size = post[dist_key].batch_shape[batch_dim]
        assert pri[dist_key].batch_shape == post[dist_key].batch_shape

        assert recon[dist_key].batch_shape \
               == recon[sample_key].shape \
