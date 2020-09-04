import torch

from code_slac.network.latent import Gaussian, ConstantGaussian
from code_slac.network.base import BaseNetwork

from diayn_seq_code_revised.networks.my_gaussian import MyGaussian


class MinVae(BaseNetwork):

    def __init__(self,
                 input_size,
                 latent_dim,
                 output_size,
                 dropout=0.,
                 hidden_sizes_enc=None,
                 hidden_sizes_dec=None,
                 device='cuda'):
        super(MinVae, self).__init__()

        if hidden_sizes_enc is None:
            self.enc = MyGaussian(
                input_dim=input_size,
                output_dim=latent_dim,
                dropout=dropout,
            )
        else:
            self.enc = MyGaussian(
                input_dim=input_size,
                output_dim=latent_dim,
                hidden_units=hidden_sizes_enc,
                dropout=dropout,
            )

        self.dec = self.create_dec(
            input_dim=latent_dim,
            output_dim=output_size,
            hidden_units=hidden_sizes_dec,
            dropout=dropout,
        )

        self.prior = ConstantGaussian(latent_dim)


        self.device = device
        self.input_size = input_size
        self.output_size = output_size

    def create_dec(
            self,
            input_dim,
            output_dim,
            hidden_units,
            dropout,
    ) -> MyGaussian:
        if hidden_units is None:
            dec = MyGaussian(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=dropout,
                std=0.1,
            )
        else:
            dec = MyGaussian(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_units=hidden_units,
                dropout=dropout,
                std=0.1,
            )

        return dec

    def sample_post(self, data) -> dict:
        """
        Args:
            data        : (N, data_dim)
        Return:
            dist        : (N, latent_dim)
            sample      : (N, latent_dim)
        """
        dist = self.enc(data)
        sample = dist.rsample()

        return {
            'dist': dist,
            'sample': sample
        }

    def sample_pri(self, batch_size) -> dict:
        dist = self.prior(torch.rand(batch_size, 1).to(self.device))
        sample = dist.sample()
        return {
            'dist': dist,
            'sample': sample
        }

    def recon(self, data, latent_post) -> dict:
        recon_dist = self.dec(latent_post['sample'])
        return {
            'dist': recon_dist,
            'sample': recon_dist.rsample(),
        }

    def forward(self, data):
        latent_post = self.sample_post(data)
        latent_pri = self.sample_pri(data.size(0))
        recon = self.recon(data, latent_post)

        return {
            'recon': recon,
            'latent_post': latent_post,
            'latent_pri': latent_pri
        }
