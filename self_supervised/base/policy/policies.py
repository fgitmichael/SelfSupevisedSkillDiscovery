import torch
from prodict import Prodict
from torch.nn import functional as F
import abc
import numpy as np

from rlkit.torch.sac.diayn.policies import SkillTanhGaussianPolicy
from rlkit.torch.core import eval_np

from self_supervised.base.network.mlp import MyMlp as Mlp

from code_slac.network.base import weights_init_xavier


class ActionMapping(Prodict):
    action: float
    agent_info: dict

    def __init__(self,
                 action: float,
                 agent_info: dict):
        super(ActionMapping, self).__init__(
            action=action,
            agent_info=agent_info
        )


# Abstract Base class
class TanhGaussianPolicy(Mlp, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self,
                 hidden_sizes,
                 obs_dim,
                 action_dim,
                 std=None,
                 initializer=weights_init_xavier,
                 **kwargs):
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            initializer=weights_init_xavier,
            hidden_activation=F.leaky_relu(torch.tensor(0.2))
            **kwargs
        )
        raise NotImplementedError

    def get_action(self,
                   obs_np: np.ndarray,
                   deterministic: bool = False) -> ActionMapping:
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return ActionMapping(
            action=actions[0, :],
            agent_info={}
        )

    def get_actions(self,
                    obs_np: np.ndarray,
                    deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    @abc.abstractmethod
    def forward(self,
                obs,
                reparameterize=True,
                deterministic=False,
                return_log_prob=False):
        raise NotImplementedError
