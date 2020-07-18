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

    def train(self,
              data: tdsscds.ModeTrainerDataMappingDiscreteSkills,
              return_post_samples: bool = False
              ) -> None:
        """
        Args:
            data:
                skills_gt           : (N, skill, S)
                obs_seq             : (N, obs_dim, S)
                skill_id            : (N, 1, S)
            return_post_samples     : not used in this method. Placeholder
                                      to avoid signature change
        """
        # Call base method
        mode_post_samples = super().train(
            tdssc.ModeTrainerDataMapping(
                obs_seq=data.obs_seq,
                skills_gt=data.skills_gt
            ),
            return_post_samples=True
        )

        self.log_mode_map(
            mode_post_samples=mode_post_samples,
            trainer_data_mapping=data
        )

    def log_mode_map(self,
                     mode_post_samples: torch.Tensor,
                     trainer_data_mapping: tdsscds.ModeTrainerDataMappingDiscreteSkills):
        """
        Logs mode_map figure to tensorboard

        Args:
            mode_post_samples      : (N, mode_dim)
        """
        batch_dim = 0
        data_dim = 1
        seq_dim = -1
        batch_size = trainer_data_mapping.obs_seq.size(batch_dim)

        assert mode_post_samples.size(batch_dim) == batch_size
        assert mode_post_samples.size(data_dim) == self.model.mode_dim

        if self.is_log(self.learn_steps):

            mode_map = self.plot_mode_map(
                global_step=self.learn_steps,
                skill_id=trainer_data_mapping.skill_id,
                mode_post_samples=mode_post_samples
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
