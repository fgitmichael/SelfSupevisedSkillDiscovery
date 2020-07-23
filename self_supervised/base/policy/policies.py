import torch
import torch.nn as nn
from prodict import Prodict
from torch.nn import functional as F
import abc
import numpy as np

from rlkit.torch.sac.diayn.policies import SkillTanhGaussianPolicy
from rlkit.torch.core import eval_np, torch_ify
from rlkit.policies.base import Policy
from rlkit.torch.distributions import TanhNormal
import rlkit.torch.pytorch_util as ptu

import self_supervised.utils.typed_dicts as td
from self_supervised.base.network.mlp import MyMlp
from rlkit.torch.networks import Mlp, identity

from code_slac.network.base import weights_init_xavier

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


# Abstract Base class
class TanhGaussianPolicy(MyMlp, Policy, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self,
                 hidden_sizes,
                 obs_dim,
                 action_dim,
                 std=None,
                 initializer=weights_init_xavier,
                 hidden_activation=nn.LeakyReLU(0.2),
                 layer_norm=False,
                 output_activation=None,
                 **kwargs):

        super().__init__(
            input_size=obs_dim,
            output_size=2 * action_dim if std is None else action_dim,
            initializer=initializer,
            hidden_activation=hidden_activation,
            hidden_sizes=hidden_sizes,
            layer_norm=layer_norm,
            output_activation=output_activation,
            **kwargs
        )

        self._dimensions = dict(
            obs_dim=obs_dim,
            action_dim=action_dim
        )

    @property
    def obs_dim(self):
        return self._dimensions['obs_dim']

    @property
    def action_dim(self):
        return self._dimensions['action_dim']

    def get_action(self,
                   obs_np: np.ndarray,
                   deterministic: bool = False) -> td.ActionMapping:
        assert obs_np.shape[-1] == self._dimensions['obs_dim']

        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        assert actions.shape == self._dimensions['action_dim']

        if len(obs_np.shape) > 1:
            assert obs_np.shape[:-1] == actions.shape[:-1]

        return td.ActionMapping(
            action=actions,
            agent_info={}
        )

    def get_actions(self,
                    obs_np: np.ndarray,
                    deterministic=False) -> np.ndarray:
        return eval_np(self, obs_np, deterministic=deterministic).action

    @abc.abstractmethod
    def forward(self,
                obs: np.ndarray,
                reparameterize: bool = True,
                deterministic: bool = False,
                return_log_prob:bool = False) -> td.ForwardReturnMapping:
        """
        Args:
            obs                 : (N, obs_dim) np-array
            reparameterize      : sample action using raparameterization trick
            deterministic       : make policy deterministic
            return_log_prob     : calculate log probability of action
        Return:
            action              : (N, action_dim) np-array
            mean                : mean of action distribution (here tanhnormal)
            log_std             : log_std of action dist
            log_prob            : log prob of action
            entropy             : entropy of policy
            std                 : std of action dist
            mean_action_log_prob: log prob of action dist mean
            pre_tanh_value      : pre tanh value
        """
        raise NotImplementedError


class TanhGaussianPolicyLogStd(TanhGaussianPolicy):

    def __init__(self,
                 **kwargs):
        """
        Args:
            std
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            initializer=weights_init_xavier,
            hidden_activation=leaky relu with 0.2
            layer_norm=False,
            output_activation=None,
        """
        super(TanhGaussianPolicyLogStd, self).__init__(
            **kwargs
        )
        kwargs = Prodict(**kwargs)

        self.log_std = None
        self.std = kwargs.std

        if self.std is not None:
            self.log_std = np.log(self.std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self,
                obs: torch.Tensor,
                reparameterize: bool = True,
                deterministic: bool = False,
                return_log_prob: bool = False) -> td.ForwardReturnMapping:
        obs_tensor = obs

        if self.std is None:
            mean_log_std_cat = MyMlp.forward(self, obs_tensor)

            assert mean_log_std_cat.size(-1) == 2 * self.action_dim
            if len(mean_log_std_cat.size()) > 1 and len(obs_tensor.shape) > 1:
                assert mean_log_std_cat.shape[:-1] == obs_tensor.shape[:-1]

            mean, log_std = torch.chunk(mean_log_std_cat,
                                        chunks=2,
                                        dim=-1)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            mean = MyMlp.forward(self, obs_tensor)
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None

        if deterministic:
            action = torch.tanh(mean)

        else:
            tanh_normal = TanhNormal(mean, std)

            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )

            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return td.ForwardReturnMapping(
            action=action,
            mean=mean,
            log_std=log_std,
            log_prob=log_prob,
            entropy=entropy,
            std=std,
            mean_action_log_prob=mean_action_log_prob,
            pre_tanh_value=pre_tanh_value
        )


class MakeDeterministic(Policy):

    def __init__(self,
                 stochastic_policy: SkillTanhGaussianPolicy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)
