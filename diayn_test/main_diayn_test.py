from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.policy.skill_policy import \
    SkillTanhGaussianPolicy, MakeDeterministic
from self_supervised.network.flatten_mlp import FlattenMlp

from self_sup_combined.utils.set_seed import set_seeds, set_env_seed

from self_sup_comb_discrete_skills.data_collector.path_collector_discrete_skills import \
    PathCollectorSelfSupervisedDiscreteSkills
from self_sup_comb_discrete_skills.memory.replay_buffer_discrete_skills import \
    SelfSupervisedEnvSequenceReplayBufferDiscreteSkills

from diayn_test.algo.diayn_trainer_seqwise import DiaynTrainerSeqwise
from diayn_test.algo.algorithm_diayn_seqwise import DiaynAlgoSeqwise
from diayn_test.utils.typed_dicts import VariantMapping
from diayn_test.utils.get_variant import parse_variant

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger


def run(variant: VariantMapping):
    ptu.set_gpu_mode(True)
    expl_env = NormalizedBoxEnvWrapper(**variant.env_kwargs)
    eval_env = NormalizedBoxEnvWrapper(**variant.env_kwargs)
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    skill_dim = variant.skill_dim

    seed = 0
    set_seeds(seed)
    set_env_seed(seed, [expl_env, eval_env])

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

    df = FlattenMlp(
        input_size=obs_dim,
        output_size=10,
        hidden_sizes=variant.hidden_layer,
        layer_norm=variant.layer_norm
    )

    policy = SkillTanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=variant.hidden_layer,
        skill_dim=skill_dim,
        layer_norm=variant.layer_norm
    )
    eval_policy = MakeDeterministic(policy)

    eval_path_collector = PathCollectorSelfSupervisedDiscreteSkills(
        env=eval_env,
        policy=eval_policy
    )
    expl_step_collector = PathCollectorSelfSupervisedDiscreteSkills(
        env=expl_env,
        policy=policy
    )

    replay_buffer = SelfSupervisedEnvSequenceReplayBufferDiscreteSkills(
        max_replay_buffer_size=variant.replay_buffer_size,
        seq_len=variant.algo_kwargs.seq_len,
        mode_dim=variant.skill_dim,
        env=expl_env
    )

    trainer = DiaynTrainerSeqwise(
        env=expl_env,
        policy=policy,

        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        df=df,

        **variant.trainer_kwargs
    )

    algorithm = DiaynAlgoSeqwise(
        trainer=trainer,

        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,

        **variant.algo_kwargs
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = parse_variant()
    setup_logger('Self-Sup-Discrete Skills', variant=variant)
    run(variant)