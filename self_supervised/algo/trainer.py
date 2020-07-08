import torch
import torch.nn as nn
import numpy as np
from typing import Iterable

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import Trainer
from rlkit.torch.networks import FlattenMlp

from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.policy.skill_policy import SkillTanhGaussianPolicy
from self_supervised.utils.typed_dicts import *
from self_supervised.utils.conversion import np_dict_to_torch
from self_supervised.loss.loss_intrin_selfsup import reconstruction_based_rewards
from self_supervised.algo.trainer_mode_latent import \
    ModeLatentTrainer, ModeLatentNetworkWithEncoder

from mode_disent_no_ssm.network.mode_model import ModeLatentNetwork




class SelfSupTrainer(Trainer):
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
                 target_entropy=None
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
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy

            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()

            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr
            )

        self.qf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0

    def train(self, data: TransitionModeMapping):
        data_torch = np_dict_to_torch(data)
        data_torch = TransitionModeMappingTorch(**data_torch)

        # Reward





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


