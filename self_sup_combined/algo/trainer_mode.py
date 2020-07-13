import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain


from self_sup_combined.network.mode_encoder import ModeEncoderSelfSupComb

import self_supervised.utils.typed_dicts as td

from mode_disent_no_ssm.utils.empty_network import Empty

from code_slac.utils import calc_kl_divergence, update_params

from mode_disent.utils.mmd import compute_mmd_tutorial


class ModeTrainer:

    def __init__(self,
                 mode_net: ModeEncoderSelfSupComb,
                 info_loss_params: td.InfoLossParamsMapping,
                 feature_extractor_net: nn.Module = Empty(),
                 lr = 0.0001
                 ):
        self.info_loss_params = info_loss_params
        self.model = mode_net
        self.feature_extractor = feature_extractor_net
        self.optim = torch.optim.Adam(
            chain(
                self.model.parameters(),
                self.feature_extractor.parameters()
            ),
            lr=lr
        )

        self.learn_steps = 0
        self.epoch = 0

    def loss(self,
             features_seq: torch.Tensor,
             skill_per_seq: torch.Tensor
             ) -> torch.Tensor:
        """
        features_seq        : (N, S, feature_dim) tensor
        skills_gt           : (N, skill_dim) skill per sequence
        """
        assert len(skill_per_seq.shape) == 2
        assert len(features_seq.shape) == 3

        mode_enc = self.model(features_seq)
        assert mode_enc.shape == skill_per_seq.shape

        # KLD
        kld = calc_kl_divergence([mode_enc['post']['dist']],
                                 [mode_enc['pri']['dist']])

        # MMD
        mmd = compute_mmd_tutorial(mode_enc['pri']['sample'],
                                   mode_enc['post']['sample'])

        # 'Reconstruction' loss
        ll = mode_enc['post']['dist'].log_prob(skill_per_seq).mean(dim=0).sum()
        mse = F.mse_loss(mode_enc['post'].loc, skill_per_seq)

        # Info VAE loss
        alpha = self.info_loss_params.alpha
        lamda = self.info_loss_params.lamda
        kld_info = (1 - alpha) * kld
        mmd_info = (alpha + lamda - 1) * mmd
        info_loss = mse + kld_info + mmd_info

        return info_loss

    def train(self,
              skills_gt: torch.Tensor,
              obs_seq: torch.Tensor,
              ) -> None:
        """
        Args:
            obs_seq      : (N, obs_dim, S) tensor
            skills_gt    : (N, skill_dim, S) tensor of skills that are all the same, since
                           this method is based on extraction of features out of the
                           features_seq and this make only sense if on the whole sequence
                           the same skill was applied
        """
        batch_dim = 0
        data_dim = 1
        seq_dim = 2
        batch_size = obs_seq.size(batch_dim)
        seq_len = obs_seq.size(seq_dim)

        assert len(obs_seq.shape) == len(skills_gt.shape) == 3
        assert obs_seq.size(batch_dim) == skills_gt.size(batch_dim)
        assert torch.all(skills_gt)
        assert torch.stack([skills_gt[:, :, 0]] * seq_len, dim=seq_dim) == skills_gt

        # In: (N, S, obs_dim) Out: (N, S, feature_dim)
        features_seq = self.feature_extractor(obs_seq.transpose(seq_dim, data_dim))
        skills = skills_gt[:, :, 0]

        loss = self.loss(
            features_seq=features_seq,
            skill_per_seq=skills_gt
        )

        # Note: Network is needed for gradient clipping, but only one model can be
        #       put in up to now, but with self.optim two networks (feat_enc and model)
        #       are being trained
        update_params(
            optim=self.optim,
            network=self.model,
            loss=loss,
            grad_clip=False, # Can't do gradient clipping with two networks
        )

    def end_episode(self, epoch):
        self.learn_steps += 1
        self.epoch = epoch










