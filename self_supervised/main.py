import torch

from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.policy.skill_policy import \
    SkillTanhGaussianPolicy, MakeDeterministic
from self_supervised.base.data_collector.data_collector import PathCollectorSelfSupervised
from self_supervised.utils.typed_dicts import VariantMapping
from self_supervised.utils.get_variant import parse_variant
from self_supervised.algo.trainer_mode_latent import ModeLatentTrainer
from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer
from self_supervised.algo.trainer import SelfSupTrainer
from self_supervised.algo.algorithm import SelfSupAlgo
from self_supervised.network.mode_latent_model import ModeLatentNetworkWithEncoder
from self_supervised.network.flatten_mlp import FlattenMlp

#from rlkit.torch.networks import FlattenMlp
import rlkit.torch.pytorch_util as ptu



def run(variant: VariantMapping):
    ptu.set_gpu_mode(True)
    expl_env = NormalizedBoxEnvWrapper(**variant.env_kwargs)
    eval_env = NormalizedBoxEnvWrapper(**variant.env_kwargs)
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    skill_dim = variant.skill_dim

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=variant.hidden_layer,
        layer_norm=variant.layer_norm

    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=variant.hidden_layer,
        layer_norm=variant.layer_norm,
    )

    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=variant.hidden_layer,
        layer_norm=variant.layer_norm,
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=variant.hidden_layer,
        layer_norm=variant.layer_norm,
    )

    mode_latent = ModeLatentNetworkWithEncoder(
        obs_dim=expl_env.env.observation_space.shape[0],
        mode_dim=skill_dim,
        representation_dim=obs_dim,
        action_dim=action_dim,
        device=ptu.device,
        **variant.mode_latent_kwargs
    )

    mode_latent_trainer = ModeLatentTrainer(
        env=expl_env,
        feature_dim=variant.mode_latent_kwargs.feature_dim,
        mode_dim=variant.skill_dim,
        mode_latent=mode_latent,
        info_loss_parms=variant.info_loss_kwargs,
        lr=0.0001,
    )

    policy = SkillTanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=variant.hidden_layer,
        skill_dim=skill_dim,
        layer_norm=variant.layer_norm
    )
    eval_policy = MakeDeterministic(policy)

    eval_path_collector = PathCollectorSelfSupervised(
        env=eval_env,
        policy=eval_policy,
    )
    expl_step_collector = PathCollectorSelfSupervised(
        env=expl_env,
        policy=policy
    )

    replay_buffer = SelfSupervisedEnvSequenceReplayBuffer(
        max_replay_buffer_size=variant.replay_buffer_size,
        seq_len=variant.seq_len,
        mode_dim=variant.skill_dim,
        env=expl_env,
    )

    trainer = SelfSupTrainer(
        env=eval_env,

        policy=policy,

        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,

        mode_latent_model=mode_latent,

        **variant.trainer_kwargs,
    )

    algo = SelfSupAlgo(
        trainer=trainer,
        mode_latent_trainer=mode_latent_trainer,

        exploration_env=expl_env,
        evaluation_env=eval_env,

        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,

        replay_buffer=replay_buffer,
        seq_len=variant.seq_len,

        **variant.algo_kwargs
    )

    algo.to(ptu.device)
    algo.train()


if __name__ == "__main__":
    variant = parse_variant()
    run(variant)


