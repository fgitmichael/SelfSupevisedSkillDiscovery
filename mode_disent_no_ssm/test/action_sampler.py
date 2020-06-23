import torch

from mode_disent.test.action_sampler import ActionSampler
from mode_disent_no_ssm.network.mode_model import ModeLatentNetwork as ModeLatentNetworkNoSSM


class ActionSamplerNoSSM(ActionSampler):

    def __init__(self,
                 mode_model: ModeLatentNetworkNoSSM,
                 device):
        self.mode_model = mode_model
        self.device = device if torch.cuda.is_available() else 'cpu'

        self._mode = None
        self._mode_next = None

    def reset(self, mode=None):
        if mode is None:
            mode_to_set = self.mode_model.sample_mode_prior(batch_size=1)['samples']
            self.set_mode(mode_to_set)
        else:
            self.set_mode(mode)

    def set_mode(self, mode):
        self._mode = mode.to(self.device)
        self._mode_next = mode.to(self.device)

    def set_mode_next(self, mode):
        self._mode_next = mode.to(self.device)

    def update_mode_to_next(self):
        self._mode = self._mode_next

    def _get_action(self,
                    mode,
                    state_rep):
        """
        Args:
            mode       : (1, mode_dim) tensor
            state_rep  : (1, state_rep_dim) tensor
        Return:
            action     : (1, action_dim) tensor
        """
        action_recon = self.mode_model.action_decoder(
            state_rep_seq=state_rep,
            mode_sample=mode
        )

        return action_recon['samples']

    def __call__(self, state_rep):
        """
        Args:
            state_rep     : (1, state_rep_dim) tensor
        """
        # Action decoder needs sequences of type (N, S, dim)
        state_rep = state_rep.unsqueeze(0)
        return self._get_action(self._mode, state_rep)
