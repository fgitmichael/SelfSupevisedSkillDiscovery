import torch

from torch.nn import functional as F

from cont_skillspace.utils.info_loss import InfoLoss


class InfoLossLatentGuided(InfoLoss):

    def _data_loss(self, data_dict):
        """
        Args:
            data_dict
                post                : (N, latent_dim) dist and samples
                recon               : (N, data_dim) dist and samples
                data                : (N, data_dim) tensor
                latent_guide        : (N, latent_dim)
                                      usage i.e. latent_dim==skill_dim
        Return:
            loss                    : scalar tensor
            log_dict                : dictionary
        """
        latent_guide = data_dict['latent_guide']
        super_ret_dict = super()._data_loss(data_dict)
        post = data_dict['post']
        data_loss_guided = F.mse_loss(post[self.dist_key].loc,
                                      latent_guide)

        loss_on_data = data_loss_guided

        log_dict = dict(
            guided_loss = data_loss_guided,
        )
        log_dict = {**log_dict,
                    **super_ret_dict['log_dict']}
        log_dict['loss_data'] = loss_on_data

        return dict(
            loss=loss_on_data,
            log_dict=log_dict
        )

    def loss(self,
             pri: dict,
             post: dict,
             recon: dict,
             data: torch.Tensor,
             latent_guide: torch.Tensor=None,
             dist_key=None,
             sample_key=None):
        """
        Args:
            ...                         : see base
            latent_guide                : (N, latent_dim)
                                          usage i.e. latent_dim==skill_dim
        """
        if latent_guide is None:
            # To avoid changing signature
            raise ValueError('Guide argument is missing')


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
                 post=post,
                 latent_guide=latent_guide)
        )

        info_loss = data_loss_dict['loss'] + \
                    latent_loss_dict['kld_info'] + \
                    latent_loss_dict['mmd_info']

        log_dict = {**latent_loss_dict['log_dict'],
                    **data_loss_dict['log_dict']}

        return info_loss, log_dict
