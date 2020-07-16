import torch
import numpy as np

from rlkit.torch.core import eval_np, torch_ify, np_ify
from rlkit.policies.base import Policy
import rlkit.torch.pytorch_util as ptu

from self_supervised.base.policy.policies import TanhGaussianPolicyLogStd
import self_supervised.utils.my_pytorch_util as my_ptu
import self_supervised.utils.typed_dicts as td

from code_slac.network.base import weights_init_xavier

class SkillTanhGaussianPolicy(TanhGaussianPolicyLogStd):

    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_sizes,
                 std=None,
                 initializer=weights_init_xavier,
                 skill_dim=2,
                 layer_norm=False,
                 **kwargs):
        self.skill_dim = skill_dim
        self.skill = None

        super().__init__(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim + self.skill_dim,
            action_dim=action_dim,
            std=std,
            initializer=initializer,
            layer_norm=layer_norm,
            **kwargs
        )

    @property
    def obs_dim(self):
        return super().obs_dim - self.skill_dim

    def get_action(self,
                   obs_np: np.ndarray,
                   skill: torch.Tensor = None,
                   deterministic: bool = False)->td.ActionMapping:
        obs_tensor = torch_ify(obs_np)

        if skill is not None:
            self._check_skill(skill)
            self.skill = skill

            assert len(self.skill.shape) == len(obs_tensor)
            if len(obs_tensor) > 1:
                assert self.skill.shape[:-1] == obs_tensor.shape[:-1]

        obs_skill_cat = torch.cat([obs_tensor, self.skill], dim=-1)

        action = self.get_skill_actions(obs_skill_cat,
                                        deterministic=deterministic)

        assert action.shape[-1] == self._dimensions['action_dim']
        if len(obs_tensor.shape) > 1:
            assert obs_tensor.shape[:-1] == action[:-1]

        return td.ActionMapping(
            action=action,
            agent_info={
                'skill': ptu.get_numpy(self.skill)
            }
        )

    def set_skill(self,
                  skill: torch.Tensor):
        """
        Args:
            skill       : (skill_dim) tensor
        """
        self._check_skill(skill)
        self.skill = skill

    def get_actions(self,
                    obs_np: np.ndarray,
                    deterministic=False) -> np.ndarray:
        # To avoid changes of signture
        raise NotImplementedError('This method should not be used in this class.'
                                  'Use method get_skill_actions instead')

    def get_skill_actions(self,
                          obs_skill_cat: torch.Tensor,
                          deterministic: bool=False
                          ) -> np.ndarray:
        action_tensor = super().forward(
            obs=obs_skill_cat,
            deterministic=deterministic
        ).action

        return ptu.get_numpy(action_tensor)

    def forward(self,
                obs: torch.Tensor,
                skill_vec=None,
                reparameterize=True,
                return_log_prob=False,
                deterministic=False) -> td.ForwardReturnMapping:
        """
        Args
            obs                     : (N, obs_dim) tensor
            skill_vec               : (N, skill_dim) tensor
            reparameterize          : bool
            return_log_prob         : bool
            deterministic           : bool
        Return:
            ForwardReturnMapping
        """
        batch_dim = 0
        data_dim = -1

        batch_size = obs.size(batch_dim)
        if skill_vec is None:
            assert len(self.skill.shape) == len(obs.shape) == 2

            skill_to_cat = torch.stack([self.skill] * batch_size, dim=0)
            obs_skill_cat = torch.cat([obs, skill_to_cat], dim=data_dim)

        else:
            assert len(skill_vec.shape) == len(obs.shape) == 2
            assert skill_vec.size(batch_dim) == obs.size(batch_dim)

            self._check_skill(skill_vec)
            obs_skill_cat = torch.cat([obs, skill_vec], dim=data_dim)

        return super().forward(
            obs=obs_skill_cat,
            reparameterize=reparameterize,
            return_log_prob=return_log_prob,
            deterministic=deterministic
        )

    def _check_skill(self,
                     skill: torch.Tensor):
        assert isinstance(skill, torch.Tensor)
        assert skill.size(-1) == self.skill_dim


class MakeDeterministic(Policy):

    def __init__(self,
                 stochastic_policy: SkillTanhGaussianPolicy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)
