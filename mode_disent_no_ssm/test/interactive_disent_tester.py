import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from mode_disent.memory.memory import MyLazyMemory
from mode_disent.env_wrappers.rlkit_wrapper import NormalizedBoxEnvForPytorch
from mode_disent.test.ia_widget_mode_setting import IaVisualization

from mode_disent_no_ssm.network.mode_model import ModeLatentNetwork as ModeLatentNetworkNoSSM
from mode_disent_no_ssm.test.action_sampler import ActionSamplerNoSSM


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

        self.viz = IaVisualization(
            fig=mode_map_fig,
            update_rate=20,
            change_mode_fun=self.mode_action_sampler.set_mode_next,
            reset_env_fun=self.env.reset
        )

        self.steps = 0
        self.episodes = 0
        self.num_episodes = num_episodes
        self.seq_len = len_sequence

    def get_feature(self, obs):
        """
        obs       : (1, observation_shape) ndarray
        """
        assert obs.shape == (1, self.env.observation_space.shape[0])

        feature = self.obs_encoder(obs)

        return feature

    def run_episode(self):
        obs = self.env.reset()
        self.env.render()

        done=False
        episode_steps = 0
        self.mode_action_sampler.update_mode_to_next()

        while not done and episode_steps < self.seq_len:
            action_tensor = self.mode_action_sampler(self.get_feature(obs))
            action = action_tensor[0].detach().cpu().numpy()

            obs, _, done = self.env.step(action)
            episode_steps += 1

            self.viz.update_plot()

        self.episodes += 1
        print(str(self.num_episodes - self.episodes) + ' episodes left')
