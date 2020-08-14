from ce_vae_test.networks.min_vae import MinVae


class VaeRegressor(MinVae):

    def forward(self,
                data,
                train=False):
        out_dict = super(VaeRegressor, self).forward(data)

        if train:
            return dict(
                recon=out_dict['recon'],
                post=out_dict['latent_post'],
                pri=out_dict['latent_pri'],
            )

        else:
            return out_dict['latent_post']
