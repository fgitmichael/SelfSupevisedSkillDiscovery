import torch
from torch.optim import Adam
import torch.nn.functional as F
import gym
from itertools import chain
import numpy as np

import rlkit.torch.pytorch_util as ptu

from code_slac.utils import calc_kl_divergence, update_params

from mode_disent.utils.mmd import compute_mmd_tutorial

from self_supervised.utils.typed_dicts import InfoLossParamsMapping
from self_supervised.network.mode_latent_model import ModeLatentNetworkWithEncoder



class ModeLatentTrainer():

    def __init__(self,
                 env: gym.Env,
                 feature_dim: int,
                 mode_dim: int,
                 info_loss_parms: InfoLossParamsMapping,
                 mode_latent: ModeLatentNetworkWithEncoder,
                 lr = 0.0001):

        self.obs_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.feature_dim = feature_dim
        self.mode_dim = mode_dim
        self.info_loss_params = info_loss_parms

        self.model = mode_latent
        self.optim = Adam(
            chain(
                self.model.parameters(),
            ),
            lr=lr)

        self.learn_steps = 0

    def train(self,
              skills: torch.Tensor,
              obs_seq: torch.Tensor,
              action_seq: torch.Tensor) -> None:
        loss = self._calc_loss(
            skills=skills,
            obs_seq=obs_seq,
            action_seq=action_seq)

        update_params(
            optim=self.optim,
            network=self.model,
            loss=loss,
        )

        self.learn_steps += 1

    def numpy_to_tensor(self, array: np.ndarray):
        return ptu.from_numpy(array)

    def _calc_loss(self,
                   skills: torch.Tensor,
                   obs_seq: torch.Tensor,
                   action_seq: torch.Tensor):
        """
        Args:
            skills             : (N, skill_dim, 1) array
            obs_seq            : (N, obs_dim, S) array
            action_seq         : (N, action_dim, S) array
        """
        state_seq = obs_seq

        assert state_seq.size(-1) == action_seq.size(-1)
        assert skills.shape[1:] == torch.Size([self.mode_dim, 1])
        if len(state_seq.size()) > 2:
            batch_size = state_seq.size(0)
            assert skills.size(0) == state_seq.size(0) == action_seq.size(0)
        else:
            # TODO: implement two-dimensional case
            raise NotImplementedError('Tensors have to be three dimensional')

        mode_post = self.model.sample_mode_posterior(obs_seq=ptu.from_numpy(obs_seq))
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
            state_rep_seq=obs_seq,
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

    def end_epoch(self, epoch):
        pass
