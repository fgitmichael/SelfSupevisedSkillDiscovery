import torch
import numpy as np
import os

from abc import ABC, abstractmethod, ABCMeta
from mode_disent.network.mode_model import ModeLatentNetwork
from mode_disent.network.dynamics_model import DynLatentNetwork
import rlkit.torch.sac.diayn



class ActionSampler(ABC):

    @abstractmethod
    def __call__(self, feature):
        """returns action"""


class ActionSamplerSeq(ActionSampler):

    def __init__(self, action_seq):
        assert len(action_seq.shape) == 2
        self.num_sequence = action_seq.size(0)
        self.action_seq = action_seq.detach().cpu()
        self._p = 0

    def _increment_p(self):
        self._p = (self._p + 1) % self.num_sequence

    def __call__(self, feature):
        action = self.action_seq[self._p].numpy()
        self._increment_p()
        return action

class ActionSamplerWithActionModel(ActionSampler):

    def __init__(self,
                 mode_model: ModeLatentNetwork,
                 dyn_model: DynLatentNetwork,
                 device,
                 mode_init=None):
        # Latent model
        self.mode_model = mode_model
        self.dyn_model = dyn_model
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Temporaray storage
        self._mode = None
        self._latent2_sample_before = None
        self._action_before = None

    def reset(self, mode=None):
        if mode is None:
            self._mode = self.mode_model.sample_mode_prior(batch_size=1)

        else:
            self._mode = mode

        self._latent2_sample_before = None
        self._action_before = None

    def _get_action(self,
                   mode,
                   feature,
                   latent2_sample_before=None,
                   action_before=None):
        """
        Args:
            mode                    : (1, mode_dim) - tensor
            feature                 : (1, feature_dim) - tensor
            latent1_sample_before   : (1, latent1_dim) - tensor
            latent2_sample_before   : (1, latent2_dim) - tensor
            action_before           : (action_dim) - tensor
        """
        if latent2_sample_before is None and action_before is None:

            latent1_dist = self.dyn_model.latent1_init_posterior(feature)
            latent1_sample = latent1_dist.rsample()

            latent2_dist = self.dyn_model.latent2_init_posterior(latent1_sample)
            latent2_sample = latent2_dist.rsample()

        else:

            latent1_dist = self.dyn_model.latent1_prior(
                [latent2_sample_before, action_before, feature])
            latent1_sample = latent1_dist.rsample()

            latent2_dist = self.dyn_model.latent2_prior(
                [latent1_sample, latent2_sample_before, action_before]
            )
            latent2_sample = latent2_dist.rsample()

        action_dist = self.mode_model.action_decoder(
            latent1_sample=latent1_sample,
            latent2_sample=latent2_sample,
            mode_sample=mode
        )
        action = action_dist.loc

        return {'action': action,
                'latent2_sample': latent2_sample}

    def __call__(self, feature):
        res = self._get_action(mode=self._mode,
                               feature=feature,
                               latent2_sample_before=self._latent2_sample_before,
                               action_before=self._action_before)

        self._latent2_sample_before = res['latent2_sample']
        self._action_before = res['action']
        self._action_before = res['action']

        return res['action']
