from code_slac.network.latent import Gaussian


class GaussianInputDimSelect(Gaussian):

    def __init__(self,
                 *args,
                 input_dim: int,
                 used_dims: tuple,
                 **kwargs):
        assert input_dim == len(used_dims)
        super().__init__(
            *args,
            input_dim=input_dim,
            **kwargs
        )
        self.used_dims = used_dims

    def forward(self, x):
        x = x[..., self.used_dims]
        return super().forward(x)
