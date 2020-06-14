import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from mode_disent.memory.memory import MyLazyMemory
from mode_disent.agent import DisentAgent
from mode_disent.network.dynamics_model import DynLatentNetwork
from mode_disent.network.mode_model import ModeLatentNetwork
from mode_disent.test.action_sampler import ActionSamplerWithActionModel
from mode_disent.test.ia_widget_mode_setting import IaVisualization
from code_slac.env.ordinary_env import OrdinaryEnvForPytorch


class InteractiveDisentTester:

    def __init__(self,
                 dyn_model: DynLatentNetwork,
                 mode_model: ModeLatentNetwork,
                 device: str,
                 env: OrdinaryEnvForPytorch,
                 mode_map_fig,
                 num_episodes: int,
                 len_sequence: int,
                 seed=0):

        self.device = device if torch.cuda.is_available() else "cpu"

        self.dyn_model = dyn_model.to(self.device)
        self.dyn_model.device = self.device
        self.dyn_model.eval()

        self.mode_model = mode_model.to(self.device)
        self.mode_model.device = self.device
        self.mode_model.eval()

        self.mode_action_sampler = ActionSamplerWithActionModel(
            mode_model=self.mode_model,
            dyn_model=self.dyn_model,
            device=self.device
        )
        self.mode_action_sampler.reset()

        self.env = env

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        self.viz = IaVisualization(
            fig=mode_map_fig,
            update_rate=20,
            change_mode_fun=self.mode_action_sampler.change_mode_on_fly)

        self.steps = 0
        self.episodes = 0
        self.num_episodes = num_episodes
        self.seq_len = len_sequence

    def get_feature(self, obs):
        # TODO: env seems to occasionally return in shape (action_dim, 1)
        #       instead of expected (1, action_dim) shape.
        #       Find out why!
        # Fix of the problem described above (dirty)
        if not obs.shape[0] == 1:
            obs = obs.T.squeeze()

        feature = self.dyn_model.state_to_feature(obs)

        return feature

    def run_episode(self):
        # Reset env
        obs = self.env.reset()
        self.env.render()
        done = False
        episode_steps = 0

        while not done and episode_steps < self.seq_len:
            # Get action
            action_tensor = self.mode_action_sampler(self.get_feature(obs))
            action = action_tensor.detach().cpu().numpy()

            # Apply action
            obs, _, done, _ = self.env.step(action)
            self.env.render()

            self.steps += self.env.action_repeat
            episode_steps += 1

            self.viz.update_plot()

        self.episodes += 1
        print(str(self.num_episodes - self.episodes) + ' episodes left')

    def run(self):
        for _ in range(self.num_episodes):
            self.run_episode()
