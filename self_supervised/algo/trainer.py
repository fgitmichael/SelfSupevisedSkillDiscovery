import torch
import torch.nn as nn
from typing import Iterable

from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.networks import FlattenMlp

from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.policy.skill_policy import SkillTanhGaussianPolicy

from mode_disent_no_ssm.network.mode_model import ModeLatentNetwork



class SelfSupTrainer(TorchTrainer):
    def __init__(self,
                 env: NormalizedBoxEnvWrapper,
                 policy: SkillTanhGaussianPolicy,
                 qf1: FlattenMlp,
                 qf2: FlattenMlp,
                 target_qf1: FlattenMlp,
                 target_qf2: FlattenMlp,
                 mode_latent_network: ModeLatentNetwork,

                 discount=0.99,
                 reward_scale=1.0,

                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 df_lr=1e-3,
                 optimizer_class=torch.optim.Adam,

                 soft_target_tau=1e-2,
                 target_update_period=1,
                 plotter=None,
                 render_eval_paths=False,

                 use_automatic_entropy_tuning=True,
                 ):
        super().__init__()

        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.mode_latent_network = mode_latent_network

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning

    @property
    def networks(self) -> Iterable[nn.Module]:
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
            mode_latent=self.mode_latent_network,
        )


