import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from typing import Dict, Tuple
import matplotlib
from matplotlib import pyplot as plt

import self_sup_combined.utils.typed_dicts as tdssc
from self_sup_combined.network.mode_encoder import ModeEncoderSelfSupComb
from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter

import self_supervised.utils.typed_dicts as td
from self_supervised.base.trainer.trainer_base import MyTrainerBaseClass
from self_supervised.base.writer.writer_base import WriterDataMapping, WriterBase

from code_slac.utils import calc_kl_divergence, update_params

from mode_disent.utils.mmd import compute_mmd_tutorial

import rlkit.torch.pytorch_util as ptu

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

    def loss(self,
             obs_seq: torch.Tensor,
             skill_per_seq: torch.Tensor
             ) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        obs_seq             : (N, S, feature_dim) tensor
        skills_gt           : (N, skill_dim) skill per sequence
        """
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

        diagnostics_scalar = {
            'kld': kld,
            'kld_info_weighted': kld_info,
            'mmd': mmd,
            'mmd_info_weighted': mmd_info,
            'mse': mse,
            'info_loss': info_loss
        }

        mode_map_data = {
            'global_step': self.learn_steps,
            'skill_gt_oh': skill_per_seq,
            'mode_post_samples': mode_enc['post']['dist'].loc
        }

        return info_loss, diagnostics_scalar, mode_map_data

    def train(self,
              data: tdssc.ModeTrainerDataMapping,
              ) -> None:
        """
        Args:
            obs_seq      : (N, obs_dim, S) tensor
            skills_gt    : (N, skill_dim, S) tensor of skills that are all the same, since
                           this method is based on extraction of features out of the
                           features_seq and this make only sense if on the whole sequence
                           the same skill was applied
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

        loss, _, _ = self.loss(
            obs_seq=obs_seq.transpose(seq_dim, data_dim),
            skill_per_seq=skills_per_seq
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

        self.learn_steps += 1

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
        super().__init__(*args, **kwargs)
        super().__init__(log_interval=log_interval,
                         writer=writer)

    def loss(self,
             *args,
             **kwargs) -> Tuple[torch.Tensor, Dict, Dict]:

        loss, diagnostics_scalar_dict, mode_map_data  = super().loss(*args, **kwargs)

        if self.learn_steps % self.log_interval == 0:

            for k, v in diagnostics_scalar_dict.items():
                self.writer.add_scalar(
                    tag=k,
                    scalar_mapping=v
                )

        return loss, {}, mode_map_data


class ModeTrainerWithDiagnosticsDiscrete(ModeTrainerWithDiagnostics):

    def __init__(self,
                 *args,
                 num_skills=10,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.mode_map_data = []
        self.num_skills = 10

    def loss(self,
             *args,
             **kwargs) -> Tuple[torch.Tensor, Dict, Dict]:

        loss, diagnostics_scalar_dict, mode_map_data  = super().loss(*args, **kwargs)

        if self.learn_steps % self.log_interval == 0:

            self.mode_map_data.append(mode_map_data)

        return loss, {}, {}

    def end_epoch(self, epoch):
        super().end_epoch(epoch)

        for mode_map in self.mode_map_data:
            fig_writer_kwargs = self.plot_mode_map(**mode_map)
            self.writer.writer.add_figure(
                **fig_writer_kwargs
            )

    def plot_mode_map(self,
                      global_step: int,
                      skills_gt: torch.Tensor,
                      mode_post_samples: torch.Tensor):
        """
        Args:
            global_step         : int
            skills_gt           : (N, skill_dim) tensor
            mode_post_samples   : (N, 2) tensor
        """
        batch_dim = 0
        data_dim = -1

        skills_gt = ptu.get_numpy(skills_gt)
        mode_post_samples = ptu.get_numpy(mode_post_samples)

        assert self.model.mode_dim == mode_post_samples.size(data_dim) == 2
        assert skills_gt.size(data_dim) == self.model.mode_dim
        assert len(skills_gt.shape) == len(mode_post_samples.shape) == 2
        assert skills_gt.size(batch_dim) == mode_post_samples.size(batch_dim)

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k',
                  'darkorange', 'gray', 'lightgreen']
        assert len(colors) <= self.num_skills

        plt.interactive(False)
        _, ax = plt.subplots()
        lim = [-3., 3.]
        ax.set_ylim(lim)
        ax.set_xlim(lim)

        for skill in range(self.num_skills):
            bool_idx = skills_gt == skill
            plt.scatter(
                mode_post_samples[bool_idx, 0],
                mode_post_samples[bool_idx, 1],
                label=str(skill),
                c=colors[skill]
            )

        ax.legend()
        ax.grid(True)
        fig = plt.gcf()

        return {
            'tag': "mode_map",
            'figure': fig,
            'global_step': global_step
        }
