# Inspired by Rlkit (https://github.com/vitchyr/rlkit)
import abc
from rlkit.samplers.data_collector import DataCollector
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(self,
                 trainer: TorchTrainer,
                 exploration_env,
                 evaluation_env,
                 exploration_data_collector: DataCollector,
                 evaluation_data_collector: DataCollector,
                 replay_buffer: ReplayBuffer):
        self.trainer = trainer

