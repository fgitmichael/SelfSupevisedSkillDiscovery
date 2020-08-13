import torch
from torch import nn

from ce_vae_test.trainer.ce_trainer import CeVaeTrainer
from diayn_no_oh.utils.hardcoded_grid_two_dim import NoohGridCreator


class MseVaeTrainer(CeVaeTrainer):

    def __init__(self,
                 *args,
                 gamma=1.,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Overwrite Loss
        self.ce_criterion = None
        self.mse_criterion = nn.MSELoss()

        self.gamma = gamma

        # Two-dimensional grid
        self.grid = torch.from_numpy(
            NoohGridCreator(
                radius_factor=1.5
            ).get_grid()
        ).to(self.device).float()

    @torch.no_grad()
    def test(self, epoch, step, data, label):
        self.vae.train(False)
        forward_return_dict = self.vae(data)
        latent_post = forward_return_dict['latent_post']
        score = forward_return_dict['recon']['sample']

        legend = ['{}'.format(tensor.detach().cpu().numpy())
                  for tensor in self.grid]
        self.write_mode_map(epoch,
                            step,
                            latent_post=latent_post,
                            label=label,
                            legend=legend)

        self.write_accuracy(step, score, label)

    def loss_data(self,
                  step,
                  forward_return_dict,
                  data,
                  label) -> torch.Tensor:
        assert len(label.shape) == 1

        targets = self.grid[label]
        mse_loss_latent = self.mse_criterion(
            forward_return_dict['latent_post']['dist'].loc,
            targets
        )
        mse_loss_recon = self.mse_criterion(
            forward_return_dict['recon']['dist'].loc,
            data
        )
        mse_loss_recon_weighted = mse_loss_recon * self.gamma

        loss_on_data = mse_loss_latent + mse_loss_recon_weighted

        prestring = "info_loss_data"
        self.log(
            step,
            {
                '{}/mse_loss_recon'.format(prestring): mse_loss_recon,
                '{}/mse_loss_recon_weighted'.format(prestring): mse_loss_recon_weighted,
                '{}/mse_loss_latent'.format(prestring): mse_loss_latent,
                '{}/loss_on_data'.format(prestring): loss_on_data
            }
        )

        return loss_on_data
