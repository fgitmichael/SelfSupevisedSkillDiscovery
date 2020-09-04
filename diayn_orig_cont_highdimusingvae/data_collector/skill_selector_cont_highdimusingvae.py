import torch

import rlkit.torch.pytorch_util as ptu

import self_supervised.utils.my_pytorch_util as my_ptu

from seqwise_cont_skillspace.data_collector.skill_selector_cont_skills import \
    SkillSelectorContinous
from ce_vae_test.networks.min_vae import MinVae


class SkillSelectorContinousHighdimusingvae(SkillSelectorContinous):

    def __init__(self,
                 *args,
                 df_vae_regressor: MinVae,
                 skill_dim: int,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dfvae = df_vae_regressor
        self._skill_dim = skill_dim

    @torch.no_grad()
    def get_random_skill(self, batch_size=1) -> torch.Tensor:
        #dist = self.skill_prior(torch.tensor([1.]))
        #sample =  dist.sample().to(ptu.device)
        #assert sample.shape == torch.Size((self.skill_prior.output_size,))

        #skill = my_ptu.eval(self.dfvae.dec, sample).loc

        #return skill.detach()
        return ptu.randn(self.skill_dim)

    def get_skill_grid(self) -> torch.Tensor:
        torch_two_dim_grid = ptu.from_numpy(self.grid)
        grid_dist = my_ptu.eval(self.dfvae.dec, torch_two_dim_grid)
        return grid_dist.loc

    @property
    def skill_dim(self):
        return self._skill_dim

