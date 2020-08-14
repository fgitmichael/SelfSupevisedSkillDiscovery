from ce_vae_test.networks.min_vae import MinVae


class VaeRegressor(MinVae):

    def forward(self,
                data,
                train=False):
        out_dict = super().forward(data)

        if train:
            return out_dict

        else:
            return out_dict['latent_post']
