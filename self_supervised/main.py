from prodict import Prodict
import gym

from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.policy.skill_policy import \
    SkillTanhGaussianPolicy, MakeDeterministic
from self_supervised.base.data_collector.data_collector import PathCollectorSelfSupervised
from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer
from self_supervised.base.utils.typed_dicts import \
    ModeLatentKwargsMapping, \
    EnvKwargsMapping, \
    TrainerKwargsMapping, \
    AlgoKwargsMapping, \
    InfoLossParamsMapping

from rlkit.torch.networks import FlattenMlp

from mode_disent_no_ssm.network.mode_model import ModeLatentNetwork




class VariantMapping(Prodict):
    algorithm: str
    version: str
    hidden_layer: list
    replay_buffer_size: int
    skill_dim: int
    env_kwargs: EnvKwargsMapping
    algo_kwargs: AlgoKwargsMapping
    trainer_kwargs: TrainerKwargsMapping

    def __init__(self,
                 algorithm: str,
                 version: str,
                 hidden_layer: list,
                 replay_buffer_size: int,
                 skill_dim: int,
                 env_kwargs: EnvKwargsMapping,
                 algo_kwargs: AlgoKwargsMapping,
                 trainer_kwargs: TrainerKwargsMapping):
        super().__init__(
            algorithm=algorithm,
            version=version,
            hidden_layer=hidden_layer,
            replay_buffer_size=replay_buffer_size,
            skill_dim=skill_dim,
            env_kwargs=env_kwargs,
            algo_kwargs=algo_kwargs,
            trainer_kwargs=trainer_kwargs
        )


def run(variant: VariantMapping):
    expl_env = NormalizedBoxEnvWrapper(**variant.env_kwargs)
    eval_env = NormalizedBoxEnvWrapper(**variant.env_kwargs)
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    skill_dim = variant.skill_dim

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=variant.hidden_layer
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=variant.hidden_layer
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=variant.hidden_layer
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=variant.hidden_layer
    )
    #policy = SkillTanhGaussianPolicy(
    #    obs_dim=obs_dim,
    #    action_dim=action_dim,
    #    skill_dim=skill_dim,
    #    hidden_sizes=variant.hidden_layer,
    #)
    #eval_policy = MakeDeterministic(policy)
    #eval_path_collector = PathCollectorSelfSupervised(
    #    env=eval_env,
    #    policy=policy,
    #)
    #expl_step_collector = PathCollectorSelfSupervised(
    #    env=expl_env,
    #    policy=policy
    #)








if __name__ == "__main__":
    env_kwargs = EnvKwargsMapping(
        gym_id='MountainCarContinuous-v0',
        action_repeat=1,
        normalize_states=True
    )

    algo_kwargs = AlgoKwargsMapping(
        num_epochs=1000
    )
    trainer_kwargs = TrainerKwargsMapping(
        discount=0.99
    )
    env_id = 'mountain'
    variant = VariantMapping(
        algorithm='Self Supervised',
        version='0.0.1',
        hidden_layer=[256, 256],
        replay_buffer_size=int(1E6),
        skill_dim=2,
        env_kwargs=env_kwargs,
        trainer_kwargs=trainer_kwargs,
        algo_kwargs=algo_kwargs,
    )

    run(variant)


