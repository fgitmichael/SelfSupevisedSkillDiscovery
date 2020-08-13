import torch
from torch import nn

from ce_vae_test.trainer.ce_trainer import CeVaeTrainer
from diayn_no_oh.utils.hardcoded_grid_two_dim import NoohGridCreator


class MseVaeTrainer(CeVaeTrainer):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Overwrite Loss
        self.ce_criterion = None
        self.mse_criterion = nn.MSELoss()

        # Two-dimensional grid
        self.grid = torch.from_numpy(
            NoohGridCreator().get_grid()
        ).to(self.device).float()

    def loss_data(self,
                  step,
                  forward_return_dict,
                  label) -> torch.Tensor:
        assert len(label.shape) == 1

        targets = self.grid[label]
        mse_loss = self.mse_criterion(
            forward_return_dict['latent_post']['dist'].loc,
            targets
        )

        loss_on_data = mse_loss

        prestring = "info_loss"
        self.log(
            step,
            {
                'mse_loss': mse_loss,
                '{}/loss_on_data'.format(prestring): loss_on_data
            }
        )

        return loss_on_data
