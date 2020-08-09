import torch

from code_slac.network.latent import Gaussian, ConstantGaussian
from code_slac.network.base import BaseNetwork



class MinVae(BaseNetwork):

    def __init__(self,
                 input_size,
                 latent_dim,
                 output_size,
                 hidden_sizes_enc=None,
                 hidden_sizes_dec=None,
                 device='cuda'):
        super(MinVae, self).__init__()

        if hidden_sizes_enc is None:
            self.enc = Gaussian(
                input_dim=input_size,
                output_dim=latent_dim,
            )
        else:
            self.enc = Gaussian(
                input_dim=input_size,
                output_dim=latent_dim,
                hidden_units=hidden_sizes_enc
            )

        self.prior = ConstantGaussian(latent_dim)

        if hidden_sizes_enc is None:
            self.dec = Gaussian(
                input_dim=latent_dim,
                output_dim=output_size,
            )
        else:
            self.dec = Gaussian(
                input_dim=input_size,
                output_dim=output_size,
                hidden_units=hidden_sizes_dec
            )

        self.device = device
        self.input_size = input_size
        self.output_size = output_size

    def sample_post(self, data):
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

    def sample_pri(self, batch_size):
        dist = self.prior(torch.rand(batch_size, 1).to(self.device))
        sample = dist.sample()
        return {
            'dist': dist,
            'sample': sample
        }

    def forward(self, data):
        latent_post = self.sample_post(data)
        latent_pri = self.sample_pri(data.size(0))
        recon = self.dec(latent_post['sample']).loc

        return {
            'recon': recon,
            'latent_post': latent_post,
            'latent_pri': latent_pri
        }
