import matplotlib.pyplot as plt
import numpy as np

from cont_skillspace_test.visualization_fun.env_viz_base import \
    EnvVisualizationGuidedBase, EnvVisualizationHduvaeBase

import rlkit.torch.pytorch_util as ptu


class EnvVisualizationPlotGuided(EnvVisualizationGuidedBase):

    def __init__(self,
                 *args,
                 plot_offset=0.05,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.fig1, self.vizualization_ax = plt.subplots()
        plt.grid()
        self.plot_offset = plot_offset

    def visualize(self):
        obs = []
        obs.append(self.obs)
        for _ in range(self.seq_len):
            a, policy_info = self.policy.get_action(self.obs)
            self.obs, r, d, env_info = self.env.step(a)
            obs.append(self.obs)
            #self.update_plot()

        obs_np = np.stack(obs, axis=1)
        self.vizualization_ax.plot(
            obs_np[0], obs_np[1])
        plt.show()

    def __del__(self):
        super().__del__()
        plt.close(self.fig1)
        print("closed fig")


class EnvVisualizationPlotHduvae(EnvVisualizationHduvaeBase):

    def __init__(self,
                 *args,
                 plot_offset,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.plot_offset = plot_offset
        self.fig1, self.vizualization_ax = plt.subplots()
        plt.grid()

    def get_skill_from_cursor_location(self, cursor_location: np.ndarray):
        latent = ptu.from_numpy(cursor_location)
        self.skill_selector.dfvae.dec.eval()
        skill = self.skill_selector.dfvae.dec(latent).loc
        return skill

    def visualize(self):
       obs = []
       obs.append(self.obs)
       for _ in range(self.seq_len):
           a, policy_info = self.policy.get_action(self.obs)
           self.obs, r, d, env_info = self.env.step(a)
           obs.append(self.obs)

       obs_np = np.stack(obs, axis=1)
       self.vizualization_ax.plot(
           obs_np[0], obs_np[1])
       plt.show()
