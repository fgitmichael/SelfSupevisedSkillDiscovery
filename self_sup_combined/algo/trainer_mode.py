import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from typing import Dict, Union, Optional
import matplotlib

import self_sup_combined.utils.typed_dicts as tdssc
from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter
from self_sup_combined.network.mode_encoder import ModeEncoderSelfSupComb

import self_supervised.utils.typed_dicts as td
from self_supervised.base.trainer.trainer_base import MyTrainerBaseClass

from code_slac.utils import calc_kl_divergence, update_params

from mode_disent.utils.mmd import compute_mmd_tutorial
from self_supervised.base.writer.writer_base import WriterBase

matplotlib.use('Agg')

class ModeTrainer(MyTrainerBaseClass):

    def __init__(self,
                 mode_net: ModeEncoderSelfSupComb,
                 info_loss_params: td.InfoLossParamsMapping,
                 lr = 0.0001
                 ):
        self.info_loss_params = info_loss_params
        self.model = mode_net

        # All parameters of the model, including the feature-encoder net
        self.optim = torch.optim.Adam(
            chain(
                self.model.parameters(),
            ),
            lr=lr
        )

        self.learn_steps = 0
        self.epoch = 0

    def _calc_loss(self,
             data: tdssc.ModeTrainerDataMapping
             ) -> Dict:
        """
        Args:
            obs_seq             : (N, S, feature_dim) tensor
            skills_gt           : (N, skill_dim) skill per sequence
        """
        skill_per_seq = data.skills_gt
        obs_seq = data.obs_seq

        assert len(skill_per_seq.shape) == 2
        assert len(obs_seq.shape) == 3

        mode_enc = self.model(obs_seq)
        assert mode_enc['post']['dist'].loc.shape == skill_per_seq.shape

        # KLD
        kld = calc_kl_divergence([mode_enc['post']['dist']],
                                 [mode_enc['pri']['dist']])

        # MMD
        mmd = compute_mmd_tutorial(mode_enc['pri']['sample'],
                                   mode_enc['post']['sample'])

        # 'Reconstruction' loss
        ll = mode_enc['post']['dist'].log_prob(skill_per_seq).mean(dim=0).sum()
        mse = F.mse_loss(mode_enc['post']['dist'].loc, skill_per_seq)

        # Info VAE loss
        alpha = self.info_loss_params.alpha
        lamda = self.info_loss_params.lamda
        kld_info = (1 - alpha) * kld
        mmd_info = (alpha + lamda - 1) * mmd
        info_loss = mse + kld_info + mmd_info

        info_loss_results = {
            'kld': kld,
            'kld_info_weighted': kld_info,
            'mmd': mmd,
            'mmd_info_weighted': mmd_info,
            'mse': mse,
            'info_loss': info_loss,
            'mode_post_samples': mode_enc['post']['dist'].loc
        }

        return info_loss_results

    def train(self,
              data: tdssc.ModeTrainerDataMapping,
              return_post_samples=False
              ) -> Optional[torch.Tensor]:
        """
        Args:
            obs_seq      : (N, obs_dim, S) tensor
            skills_gt    : (N, skill_dim, S) tensor of skills that are all the same,
                           since this method is based on extraction of features out
                           of the features_seq and this make only sense if on the
                           whole sequence the same skill was applied
        """
        skills_gt = data.skills_gt
        obs_seq = data.obs_seq

        batch_dim = 0
        data_dim = 1
        seq_dim = 2
        batch_size = obs_seq.size(batch_dim)
        seq_len = obs_seq.size(seq_dim)

        assert len(obs_seq.shape) == len(skills_gt.shape) == 3
        assert obs_seq.size(batch_dim) == skills_gt.size(batch_dim)
        assert torch.all(torch.stack([skills_gt[:, :, 0]] * seq_len, dim=seq_dim)
                         == skills_gt)

        skills_per_seq = skills_gt[:, :, 0]

        loss_results  = self._calc_loss(
            tdssc.ModeTrainerDataMapping(
                obs_seq=obs_seq.transpose(seq_dim, data_dim),
                skills_gt=skills_per_seq
            )
        )
        mode_post_samples = loss_results.pop('mode_post_samples')

        self.log_loss_results(loss_result=loss_results)

        loss = loss_results['info_loss']

        # Note: Network is needed for gradient clipping, but only one model can be
        #       put in up to now, but with self.optim two networks (feat_enc and model)
        #       are being trained
        update_params(
            optim=self.optim,
            network=self.model,
            loss=loss,
            grad_clip=False, # Can't do gradient clipping with two networks
        )

        self.learn_steps += 1

        if return_post_samples:
            return mode_post_samples

    def log_loss_results(self, loss_result: dict):
        pass

    def end_epoch(self, epoch):
        self.epoch = epoch

    def get_snapshot(self):
        return {}

    def get_diagnostics(self):
        return {}

    @property
    def networks(self) -> Dict[str, nn.Module]:
        return dict(
            mode_encoder=self.model
        )


class ModeTrainerWithDiagnostics(
    ModeTrainer,
    DiagnosticsWriter):

    def __init__(self,
                 *args,
                 log_interval,
                 writer: WriterBase,
                 **kwargs
                 ):
        ModeTrainer.__init__(self,
                             *args,
                             **kwargs)
        DiagnosticsWriter.__init__(self,
                                   log_interval=log_interval,
                                   writer=writer)

    def log_loss_results(self, loss_result: dict):

        if self.is_log(self.learn_steps):
            for k, v in loss_result.items():
                self.writer.writer.add_scalar(
                    tag="log_loss_results/{}".format(k),
                    scalar_value=v,
                    global_step=self.learn_steps
                )