from netwerk
class EncoderStateRep(BaseNetwork):

    def __init__(self,
                 input_dim,
                 output_dim,
                 leaky_slope=0.2):
        super(EncoderStateRep, self).__init__()

        self.net = create_linear_network(input_dim,
                                         output_dim,
                                         hidden_units=[24])

    def forward(self, x):
        return self.net(x)