import torch
from operator import itemgetter
from torch.nn import functional as F

from diayn_original_cont.trainer.diayn_cont_trainer import DIAYNTrainerCont


class DIAYNTrainerContHighdimusingvae(DIAYNTrainerCont):

    def _df_loss_intrinsic_reward(self,
                                  skills,
                                  next_obs):
        """
        Args:
            next_obs            : (N, obs_dim)
            skills              : (N, skill_dim)
        """
        assert len(next_obs.shape) == len(skills.shape) == 2
        batch_dim = 0
        data_dim = -1

        z_hat = skills
        train_dict = self.df(next_obs, train=True)

        skills_post_dist = train_dict['recon']['dist']
        assert skills_post_dist.batch_shape == skills.shape
        rewards = torch.sum(
            skills_post_dist.log_prob(skills),
            dim=data_dim,
            keepdim=True,
        )
        assert rewards.shape == torch.Size((next_obs.size(0), 1))

        vae_ret_dict = self.df(next_obs, train=True)

        info_loss, log_dict = self.info_loss_fun(
            **vae_ret_dict,
            data=skills,
        )

        return dict(
            df_loss=info_loss,
            rewards=rewards,
            pred_skill=vae_ret_dict['recon']['dist'].loc,
            log_dict=log_dict,
        )
