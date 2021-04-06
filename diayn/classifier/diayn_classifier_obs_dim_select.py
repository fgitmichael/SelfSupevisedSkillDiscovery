from rlkit.torch.networks import FlattenMlp


class FlattenMlpInputDimSelect(FlattenMlp):

    def __init__(self,
                 *args,
                 input_size: int,
                 used_dims: tuple,
                 **kwargs):
        assert len(used_dims) == input_size
        super().__init__(
            *args,
            input_size=input_size,
            **kwargs
        )
        self.used_dims = used_dims

    def forward(self, *inputs, **kwargs):
        assert len(inputs) == 1
        input = inputs[0][..., self.used_dims]
        return super().forward(input, **kwargs)
