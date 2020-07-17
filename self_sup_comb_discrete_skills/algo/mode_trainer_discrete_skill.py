from typing import Tuple, Dict

import torch
from matplotlib import pyplot as plt
from rlkit.torch import pytorch_util as ptu

from self_sup_combined.algo.trainer_mode import ModeTrainerWithDiagnostics


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