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
                dist                : (N, ..., latent_dim) distribution
                sample              : (N, ..., latent_dim) samples
            post
                dist                : (N, ..., latent_dim) distribution
                sample              : (N, ..., latent_dim) samples
            recon
                dist                : (N, ..., data_dim) distribution
                sample              : (N, ..., data_dim) samples
                                      (for gaussians typically loc instead of samples)
            data                    : (N, ..., data_dim) tensor of original data

        Return:
            loss                    : scalar tensor
            log_values
                kld                 : scalar tensor
                mmd                 : scalar tensor
                mse                 : scalar tensor
                kld_info            : scalar tensor
                mmd_info            : scalar tensor
                loss_latent         : scalar tensor
                loss_data           : scalar tensor
                info_loss           : scalar tensor
        """
        if dist_key is None:
            dist_key = 'dist'
        if sample_key is None:
            sample_key = 'sample'

        batch_dim = 0
        batch_size = post[dist_key].batch_shape[batch_dim]
        assert pri[dist_key].batch_shape == post[dist_key].batch_shape

        assert recon[dist_key].batch_shape \
               == recon[sample_key].shape \
               == data.shape

        kld = calc_kl_divergence([post[dist_key]],
                                 [pri[dist_key]])

        mmd = compute_mmd_tutorial(pri[sample_key],
                                   post[sample_key])

        mse = F.mse_loss(recon[sample_key],
                         data)

        alpha = self.alpha
        lamda = self.lamda
        kld_info = (1 - alpha) * kld
        mmd_info = (alpha + lamda - 1) * mmd
        info_loss = mse + kld_info + mmd_info

        loss_latent = mmd_info + kld_info
        loss_data = mse

        log_dict = dict(
            kld=kld,
            mmd=mmd,
            mse=mse,
            kld_info=kld_info,
            mmd_info=mmd_info,
            loss_latent=loss_latent,
            loss_data=loss_data,
            info_loss=info_loss,
        )

        return info_loss, log_dict
