from typing import Union
import torch
from torch import nn
from torch import optim

from diayn_original_tb.policies.self_sup_policy_wrapper import \
    RlkitWrapperForMySkillPolicy, \
    MakeDeterministicMyPolicyWrapper
from diayn_original_tb.policies.diayn_policy_extension import \
    SkillTanhGaussianPolicyExtension, \
    MakeDeterministicExtension

from rlkit.torch.distributions import TanhNormal


class ActionLogpropCalculator(object):

    def __init__(self,
                 policy: Union[
                    RlkitWrapperForMySkillPolicy,
                    MakeDeterministicMyPolicyWrapper,
                    SkillTanhGaussianPolicyExtension,
                    MakeDeterministicExtension]
                 ):
        self.policy = policy

    @torch.no_grad()
    def action_log_prob(self,
                        action: torch.Tensor,
                        obs: torch.Tensor,
                        skill_vec: torch.Tensor,
                        ):
        """
        Args
            action        : (N, action_dim)
            obs           : (N, obs_dim)
            skill_vec     : (N, skill_dim)
        Return
            log_prob      : log p(a | obs, skill_vec)
        """
        mean, std = self._get_mean_std(
            obs=obs,
            skill_vec=skill_vec
           )

        # Reconstuction tanh normal distribution
        tanh_normal = TanhNormal(mean, std)

        return tanh_normal.log_prob(action)

    def _get_mean_std(self,
                      obs: torch.Tensor,
                      skill_vec: torch.Tensor,
                      ):
        (_,
         mean,
         _,
         _,
         _,
         std,
         _,
         _) = self.policy(
            obs=obs,
            skill_vec=skill_vec,
            reparameterize=False,
            return_log_prob=False
        )

        return mean, std
