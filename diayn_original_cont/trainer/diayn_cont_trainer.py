import torch

from diayn_original_cont.networks.vae_regressor import VaeRegressor
from diayn_original_cont.trainer.info_loss_min_vae import InfoLossLatentGuided

from diayn_with_rnn_classifier.trainer.diayn_trainer_modularized import \
    DIAYNTrainerModularized


class DIAYNTrainerCont(DIAYNTrainerModularized):

    def __init__(self,
                 *args,
                 info_loss_fun: InfoLossLatentGuided.loss,
                 skill_vae: VaeRegressor,
                 **kwargs):
        super(DIAYNTrainerCont, self).__init__(*args, **kwargs)
        self.info_loss_fun = info_loss_fun
        self.skill_vae = skill_vae

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
        train_dict = self.df(next_obs)

        latent_post_dist = train_dict['latent_post']['dist']
        assert latent_post_dist.batch_shape == skills.shape
        rewards = torch.sum(
            latent_post_dist.log_prob(skills),
            dim=data_dim
        )
        assert rewards.shape == torch.Size((next_obs.size(0), 1))

        vae_ret_dict = self.skill_vae(next_obs, train=True)
        info_loss, log_dict = self.info_loss_fun(
            **vae_ret_dict,
            data=next_obs,
            latent_guide=skills
        )

        return dict(
            df_loss=info_loss,
            rewards=rewards,
            pred_skill=vae_ret_dict['post']['dist'].loc,
        )
