import abc
import gym


from rlkit.samplers.data_collector import DataCollector
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.core.rl_algorithm import BaseRLAlgorithm

from self_supervised.base.data_collector.data_collector import PathCollectorSelfSupervised
from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer
from self_supervised.algo.trainer import SelfSupTrainer


class BaseRLAlgorithmSelfSup(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(self,
                 trainer: SelfSupTrainer,
                 exploration_env: gym.Env,
                 evaluation_env: gym.Env,
                 exploration_data_collector: PathCollectorSelfSupervised,
                 evaluation_data_collector: PathCollectorSelfSupervised,
                 replay_buffer: SelfSupervisedEnvSequenceReplayBuffer):
        super().__init__(
            trainer=trainer,
            exploration_env=exploration_env,
            evaluation_env=evaluation_env,
            exploration_data_collector=exploration_data_collector,
            evaluation_data_collector=evaluation_data_collector,
            replay_buffer=replay_buffer
        )

        # Reassign to get type hints
        self.trainer = trainer
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
