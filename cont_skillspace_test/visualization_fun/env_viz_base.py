import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import abc
from cont_skillspace_test.widget.ia_widget import IaVisualization

import rlkit.torch.pytorch_util as ptu

class EnvVisualizationBase(metaclass=abc.ABCMeta):
    def __init__(self,
                 env: gym.Env,
                 policy: torch.nn.Module,
                 ):
        self.env = env
        self.policy = policy

        self.visualization = IaVisualization(
            reset_fun=self.reset,
            set_next_skill_fun=self.set_skill,
        )

        self.obs = None

    def set_skill(self, cursor_location: np.ndarray):
        latent = ptu.from_numpy(cursor_location)
        skill = self.get_skill_from_cursor_location(cursor_location)
        self.policy.skill = skill

    @abc.abstractmethod
    def get_skill_from_cursor_location(self, cursor_location: np.ndarray):
        raise NotImplementedError

    def reset(self):
        self.obs = self.env.reset()
        self.env.render()

    @abc.abstractmethod
    def visualize(self):
        raise NotImplementedError
        #while True:
        #    a, policy_info = self.policy.get_action(self.obs)
        #    self.obs, r, d, env_info = self.env.step(a)
        #    self.env.render()

    def update_plot(self):
        plt.pause(0.001)

    def run(self):
        while True:
            self.update_plot()

    def __del__(self):
        self.env.close()
