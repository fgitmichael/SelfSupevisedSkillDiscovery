from prodict import Prodict
import gym

from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.policy.skill_policy import \
    SkillTanhGaussianPolicy, MakeDeterministic
from self_supervised.base.data_collector.data_collector import PathCollectorSelfSupervised
from self_supervised.memory.self_sup_replay_buffer import SelfSupervisedEnvSequenceReplayBuffer

from rlkit.torch.networks import FlattenMlp

from


class AlgoKwargsMapping(Prodict):
    num_epochs: int
    def __init__(self,
                 num_epochs: int):
        super().__init__(
            num_epochs=num_epochs
        )


class TrainerKwargsMapping(Prodict):
    discount: float
    def __init__(self,
                 discount: float):
        super().__init__(
            discount=discount
        )


class EnvKwargsMapping(Prodict):
    gym_id: str
    action_repeat: int
    normalize_states: bool

    def __init__(self,
                 gym_id: str,
                 action_repeat: int,
                 normalize_states: bool):
        super().__init__(
            gym_id=gym_id,
            action_repeat=action_repeat,
            normalize_states=normalize_states
        )

    def init(self):
        self.normalize_states = True


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
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = PathCollectorSelfSupervised(
        env=eval_env,
        policy=policy,
    )
    expl_step_collector = PathCollectorSelfSupervised(
        env=expl_env,
        policy=policy
    )








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


