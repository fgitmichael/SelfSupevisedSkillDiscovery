from typing import Tuple, Dict
import numpy as np

import torch
from matplotlib import pyplot as plt
from rlkit.torch import pytorch_util as ptu

from self_sup_combined.algo.trainer_mode import ModeTrainerWithDiagnostics

import self_sup_combined.utils.typed_dicts as tdssc

import self_sup_comb_discrete_skills.utils.typed_dicts as tdsscds


class ModeTrainerWithDiagnosticsDiscrete(ModeTrainerWithDiagnostics):

    def __init__(self,
                 *args,
                 num_skills=10,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.mode_map_data = []
        self.num_skills = num_skills

    def log_loss_results(self, data: dict, trainer_data_mapping=None):
        assert type(trainer_data_mapping) \
            == tdsscds.ModeTrainerDataMappingDiscreteSkills

        super().log_loss_results(data=data)

        if self.is_log(self.learn_steps):
            mode_map = self.plot_mode_map(
                global_step=self.learn_steps,
                skill_id=trainer_data_mapping.skill_id,
                mode_post_samples=data['mode_post_samples']
            )
            self.writer.writer.add_figure(**mode_map)

    def plot_mode_map(self,
                      global_step: int,
                      skill_id: torch.Tensor,
                      mode_post_samples: torch.Tensor):
        """
        Args:
            global_step         : int
            skill_id            : (N, skill_dim) tensor
            mode_post_samples   : (N, 1) tensor
        """
        batch_dim = 0
        data_dim = -1

        skill_id = ptu.get_numpy(skill_id)
        mode_post_samples = ptu.get_numpy(mode_post_samples)

        assert self.model.mode_dim == mode_post_samples.size(data_dim) == 2
        assert skill_id.size(data_dim) == 1
        assert len(skill_id.shape) == len(mode_post_samples.shape) == 2
        assert skill_id.size(batch_dim) == mode_post_samples.size(batch_dim)

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k',
                  'darkorange', 'gray', 'lightgreen']
        assert len(colors) <= self.num_skills

        plt.interactive(False)
        _, ax = plt.subplots()
        lim = [-3., 3.]
        ax.set_ylim(lim)
        ax.set_xlim(lim)

        for skill in range(self.num_skills):
            bool_idx = skill_id == skill
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
            'tag': "mode_map (Note: with own loginterval, not epochwise)",
            'figure': fig,
            'global_step': global_step
        }

