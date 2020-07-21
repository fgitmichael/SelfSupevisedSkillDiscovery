import abc
import gym
from typing import Union

from diayn_test.algo.diayn_trainer_seqwise import DiaynTrainerSeqwise

from self_sup_comb_discrete_skills.data_collector.path_collector_discrete_skills import \
    PathCollectorSelfSupervisedDiscreteSkills
from self_sup_comb_discrete_skills.memory.replay_buffer_discrete_skills import \
    SelfSupervisedEnvSequenceReplayBufferDiscreteSkills

from rlkit.samplers.data_collector import DataCollector
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.core.rl_algorithm import BaseRLAlgorithm

from self_supervised.base.data_collector.data_collector import PathCollectorSelfSupervised
from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer
from self_supervised.base.trainer.trainer_base import Trainer
from self_supervised.algo.trainer import SelfSupTrainer

from self_sup_combined.algo.trainer_sac import SelfSupCombSACTrainer


class BaseRLAlgorithmSelfSup(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(self,
                 trainer: Union[Trainer,
                                SelfSupCombSACTrainer,
                                SelfSupTrainer,
                                DiaynTrainerSeqwise],
                 exploration_env: gym.Env,
                 evaluation_env: gym.Env,
                 exploration_data_collector: Union[
                     PathCollectorSelfSupervised,
                     PathCollectorSelfSupervisedDiscreteSkills],
                 evaluation_data_collector: Union[
                     PathCollectorSelfSupervised,
                     PathCollectorSelfSupervisedDiscreteSkills],
                 replay_buffer: Union[
                     SelfSupervisedEnvSequenceReplayBuffer,
                     SelfSupervisedEnvSequenceReplayBufferDiscreteSkills]
                 ):
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
