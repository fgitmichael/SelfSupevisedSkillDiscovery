import matplotlib.pyplot as plt
import numpy as np

from cont_skillspace_test.visualization_fun.env_viz_base import \
    EnvVisualizationBase

import rlkit.torch.pytorch_util as ptu

from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper


class EnvVisualizationPlotGuided(EnvVisualizationBase):

    def __init__(self,
                 *args,
                 seq_len,
                 plot_offset=0.05,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.fig1, self.vizualization_ax = plt.subplots()
        plt.grid()
        self.plot_offset = plot_offset

    def get_skill_from_cursor_location(self, cursor_location: np.ndarray):
        return ptu.from_numpy(cursor_location)

    def reset(self):
        self.obs = self.env.reset()

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

    def set_skill(self, cursor_location: np.ndarray):
        super().set_skill(cursor_location)
        self.visualize()

    def __del__(self):
        plt.close(self.fig1)
        print("closed fig")


class EnvVisualizationPlotHduvae(EnvVisualizationPlotGuided):
    def __init__(self,
                 *args,
                 skill_selector,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.skill_selector = skill_selector


    def get_skill_from_cursor_location(self, cursor_location: np.ndarray):
        latent = ptu.from_numpy(cursor_location)
        self.skill_selector.dfvae.dec.eval()
        skill = self.skill_selector.dfvae.dec(latent).loc
        return skill
