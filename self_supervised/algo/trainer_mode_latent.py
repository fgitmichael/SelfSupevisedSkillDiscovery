import torch
from torch.optim import Adam
import torch.nn.functional as F
import gym
from itertools import chain


from code_slac.utils import calc_kl_divergence, update_params

from mode_disent_no_ssm.network.mode_model import ModeLatentNetwork
from mode_disent_no_ssm.utils.empty_network import Empty

from mode_disent.utils.mmd import compute_mmd_tutorial

from self_supervised.utils.typed_dicts import InfoLossParamsMapping



class ModeLatentTrainer():

    def __init__(self,
                 env: gym.Env,
                 feature_dim: int,
                 mode_dim: int,
                 obs_encoder: torch.nn.Module,
                 info_loss_parms: InfoLossParamsMapping,
                 mode_latent: ModeLatentNetwork,
                 lr = 0.0001):

        self.obs_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.feature_dim = feature_dim
        self.mode_dim = mode_dim
        self.info_loss_params = info_loss_parms

        self.model = mode_latent
        self.obs_encoder = obs_encoder
        self.optim = Adam(
            chain(
                self.model.parameters(),
                self.obs_encoder.parameters()
            ),
            lr=lr)

        self.learn_steps = 0

    def train(self,*args, **kwargs):
        loss = self._calc_loss(*args, **kwargs)
        update_params(
            optim=self.optim,
            network=self.model,
            loss=loss,
        )
        self.learn_steps += 1

    def _calc_loss(self,
                   skills: torch.Tensor,
                   state_seq: torch.Tensor,
                   action_seq: torch.Tensor):
        """
        Args:
            skills             : (N, skill_dim, 1) tensor
            state_seq          : (N, obs_dim, S) tensor
            action_seq         : (N, action_dim, S) tensor
        """
        seq_len = state_seq.size(-1)

        assert state_seq.size(-1) == action_seq.size(-1)
        assert skills.shape[1:] == torch.Size([self.mode_dim, 1])
        if len(state_seq.size()) > 2:
            batch_size = state_seq.size(0)
            assert skills.size(0) == state_seq.size(0) == action_seq.size(0)
        else:
            # TODO: implement two-dimensional case
            raise NotImplementedError('Tensors have to be three dimensional')

        features_seq = self.obs_encoder(state_seq)

        mode_post = self.model.sample_mode_posterior(features_seq=features_seq)
        mode_pri = self.model.sample_mode_prior(batch_size)

        kld = calc_kl_divergence(
            [mode_post['dists']],
            [mode_pri['dists']]
        )

        mmd = compute_mmd_tutorial(
            mode_pri['samples'],
            mode_post['samples']
        )

        actions_seq_recon = self.model.action_decoder(
            state_rep_seq=features_seq,
            mode_sample=mode_post['samples']
        )

        ll = actions_seq_recon['dists'].log_prob(action_seq).mean(dim=0).sum()
        mse = F.mse_loss(actions_seq_recon['samples'], action_seq)

        # Info Loss
        alpha = self.info_loss_params.alpha
        lamda = self.info_loss_params.lamda
        kld_info = (1 - alpha) * kld
        mmd_info = (alpha + lamda - 1) * mmd
        info_loss = mse + kld_info + mmd_info

        return info_loss
