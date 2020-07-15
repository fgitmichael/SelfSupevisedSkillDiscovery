import numpy as np
from tqdm import tqdm
from typing import Dict
import torch

from self_supervised.base.data_collector.data_collector import \
    PathCollectorSelfSupervised
from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer
from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.base.algo.algo_base import BaseRLAlgorithmSelfSup

from self_sup_combined.algo.trainer import SelfSupCombSACTrainer
from self_sup_combined.algo.trainer_mode import ModeTrainer

import rlkit.torch.pytorch_util as ptu


class SelfSupCombAlgo(BaseRLAlgorithmSelfSup):

    def __init__(self,
                 sac_trainer: SelfSupCombSACTrainer,
                 mode_trainer: ModeTrainer,

                 exploration_env: NormalizedBoxEnvWrapper,
                 evaluation_env: NormalizedBoxEnvWrapper,

                 exploration_data_collector: PathCollectorSelfSupervised,
                 evaluation_data_collector: PathCollectorSelfSupervised,

                 replay_buffer: SelfSupervisedEnvSequenceReplayBuffer,

                 batch_size: int,
                 seq_len: int,
                 num_epochs: int,

                 num_eval_steps_per_epoch: int,
                 num_trains_per_expl_step: int,
                 num_train_loops_per_epoch: int = 1,
                 min_num_steps_before_training: int = 0,
                 ):
        super().__init__(
            trainer=sac_trainer,
            exploration_env=exploration_env,
            evaluation_env=evaluation_env,
            exploration_data_collector=exploration_data_collector,
            evaluation_data_collector=evaluation_data_collector,
            replay_buffer=replay_buffer
        )

        self.mode_trainer = mode_trainer

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_train_loops_per_epoch
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.min_num_steps_before_training = min_num_steps_before_training
        self.num_trains_per_expl_seq = num_trains_per_expl_step

        self.policy = self.trainer.policy

    @property
    def networks(self) -> Dict[str, torch.nn.Module]:
        return dict(
            **self.trainer.networks,
            **self.mode_trainer.networks
        )

    def training_mode(self, mode):
        for net in self.networks.values():
            net.train(mode)

    def to(self, device):
        for net in self.trainer.networks.values():
            net.to(device)