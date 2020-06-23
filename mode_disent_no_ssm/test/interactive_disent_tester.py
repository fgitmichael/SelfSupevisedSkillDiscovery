import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

from mode_disent.memory.memory import MyLazyMemory
from mode_disent.env_wrappers.rlkit_wrapper import NormalizedBoxEnvForPytorch

from mode_disent_no_ssm.network.mode_model import ModeLatentNetwork as ModeLatentNetworkNoSSM
from mode_disent_no_ssm.test.action_sampler import ActionSamplerNoSSM
from mode_disent_no_ssm.test.ia_widget_mode_setting import IaVisualizationNoSSM


class InteractiveDisentTester:

    def __init__(self,
                 mode_model: ModeLatentNetworkNoSSM,
                 obs_encoder,
                 device,
                 env: NormalizedBoxEnvForPytorch,
                 mode_map_fig,
                 num_episodes,
                 len_sequence,
                 seed=1):

        self.device = device if torch.cuda.is_available() else "cpu"

        self.obs_encoder = obs_encoder
        self.obs_encoder.eval()
        self.mode_model = mode_model.to(self.device)
        self.mode_model.eval()

        self.mode_action_sampler = ActionSamplerNoSSM(
            mode_model=self.mode_model,
            device=self.device
        )
        self.mode_action_sampler.reset()

        self.env = env

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        self.viz = IaVisualizationNoSSM(
            fig=mode_map_fig,
            update_rate=20,
            change_mode_fun=self.mode_action_sampler.set_mode,
            reset_env_fun=self.env.reset
        )

        self.steps = 0
        self.episodes = 0
        self.num_episodes = num_episodes
        self.seq_len = len_sequence if len_sequence is not None else sys.maxsize

    def get_feature(self, obs):
        """
        Args:
            obs       : (1, observation_shape) ndarray
        Return:
            feature   : (1, feature_dim) tensor
        """
        assert obs.shape == self.env.observation_space.shape

        obs_tensor = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        feature = self.obs_encoder(obs_tensor)

        return feature

    def run_episode(self):
        obs = self.env.reset()
        self.env.render()

        done=False
        episode_steps = 0
        self.mode_action_sampler.update_mode_to_next()
        while not done and episode_steps < self.seq_len:
            action_tensor = self.mode_action_sampler(self.get_feature(obs).float())
            action = action_tensor.squeeze().detach().cpu().numpy()

            obs, _, done, _ = self.env.step(action)
            self.env.render()
            episode_steps += 1

            self.viz.update_plot()

            if self.viz.is_should_stop():
                break

        if done:
            print('cause of done')

        self.episodes += 1
        print(str(self.num_episodes - self.episodes) + ' episodes left')

    def run(self):
        with torch.no_grad():
            for _ in range(self.num_episodes):
                self.run_episode()

                while not self.viz.is_should_start() and self.viz.is_should_stop():
                    self.viz.update_plot()
