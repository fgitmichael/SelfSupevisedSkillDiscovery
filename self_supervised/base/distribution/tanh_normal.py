import torch
from torch.distributions.normal import Normal

class TanhNormal(Normal):

    def __init__(self,
                 loc,
                 scale):
        super(TanhNormal, self).__init__(loc, scale)

    def sample(self, sample_shape=torch.Size()):
        z = self.sample(sample_shape)
        return torch.tanh(z)

    def rsample(self, sample_shape=torch.Size()):
        z = self.rsample(sample_shape)
        return torch.tanh(z)

    def log_prob(self, value):
        pre_tanh_value = self._inv_tanh(value)
        return self.log_prob(pre_tanh_value)
        #TODO: in rlkits tanh-normal class torch.log(1-value*value + eps)
        #      is substracted. Find out why

    def _inv_tanh(self, value):
        eps = 1e-7
        return torch.log(
            (1 + value + eps) / (1 - value + eps)
        ) / 2

