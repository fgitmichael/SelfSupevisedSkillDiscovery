import torch

from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.networks import FlattenMlp

from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.policy.skill_policy import SkillTanhGaussianPolicy
from self_supervised.algo.trainer_mode_latent import ModeLatentTrainer



class SelfSupTrainer(TorchTrainer):
    def __init__(self,
                 env: NormalizedBoxEnvWrapper,
                 policy: SkillTanhGaussianPolicy,
                 gf1: FlattenMlp,
                 qf2: FlattenMlp,
                 target_qf1: FlattenMlp,
                 target_qf2: FlattenMlp,
                 mode_latent_trainer: ModeLatentTrainer,

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
        pass
