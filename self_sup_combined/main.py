from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.policy.skill_policy import \
    SkillTanhGaussianPolicy, MakeDeterministic
from self_supervised.base.data_collector.data_collector import \
    PathCollectorSelfSupervised
from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer
from self_supervised.network.flatten_mlp import FlattenMlp
from self_supervised.base.network.mlp import MyMlp

from self_sup_combined.network.mode_encoder import ModeEncoderSelfSupComb
from self_sup_combined.utils.get_variant import parse_variant
from self_sup_combined.utils.typed_dicts import VariantMapping
from self_sup_combined.algo.trainer import SelfSupCombTrainer
from self_sup_combined.loss.mode_likelihood_based_reward import \
    ReconstructionLikelyhoodBasedRewards

from mode_disent_no_ssm.utils.empty_network import Empty

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

    feature_dim = variant.mode_latent_kwargs.feature_dim
    if obs_dim == feature_dim:
        obs_encoder = Empty()
    else:
        obs_encoder = MyMlp(
            input_size=obs_dim,
            output_size=feature_dim
        )

    mode_encoder = ModeEncoderSelfSupComb(
        obs_encoder=obs_encoder,
        mode_dim=variant.skill_dim,
        **variant.mode_encoder_kwargs
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
        policy=eval_policy
    )
    expl_step_collector = PathCollectorSelfSupervised(
        env=expl_env,
        policy=policy
    )

    replay_buffer = SelfSupervisedEnvSequenceReplayBuffer(
        max_replay_buffer_size=variant.replay_buffer_size,
        seq_len=variant.seq_len,
        mode_dim=variant.skill_dim,
        env=expl_env
    )

    reward_calculator = ReconstructionLikelyhoodBasedRewards(
        skill_policy=policy,
        mode_encoder=mode_encoder
    )

    trainer = SelfSupCombTrainer(
        env=expl_env,
        policy=policy,

        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,

        mode_encoder=mode_encoder,
        intrinsic_reward_calculator=reward_calculator,

        **variant.trainer_kwargs
    )


if __name__ == "__main__":
    # TODO: Set seeds
    variant = parse_variant()
    run(variant)