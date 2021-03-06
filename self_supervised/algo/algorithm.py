import numpy as np
from tqdm import tqdm

from self_supervised.algo.trainer import SelfSupTrainer
from self_supervised.base.data_collector.data_collector import \
    PathCollectorSelfSupervised
from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer
from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.base.algo.algo_base import BaseRLAlgorithmSelfSup
from self_supervised.algo.trainer_mode_latent import ModeLatentTrainer

import rlkit.torch.pytorch_util as ptu


class SelfSupAlgo(BaseRLAlgorithmSelfSup):

    def __init__(self,
                 trainer: SelfSupTrainer,
                 mode_latent_trainer: ModeLatentTrainer,

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
            trainer=trainer,
            exploration_env=exploration_env,
            evaluation_env=evaluation_env,
            exploration_data_collector=exploration_data_collector,
            evaluation_data_collector=evaluation_data_collector,
            replay_buffer=replay_buffer
        )
        self.mode_latent_trainer = mode_latent_trainer

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.min_num_steps_before_training = min_num_steps_before_training
        self.num_trains_per_expl_seq = num_trains_per_expl_step

        self.policy = trainer.policy

    def _train(self):
        self.training_mode(False)

        # Collect first steps with the untrained gaussian policy
        if self.min_num_steps_before_training > 0:
            self.expl_data_collector.collect_new_paths(
                seq_len=self.seq_len,
                num_seqs=int(np.ceil(self.min_num_steps_before_training/self.seq_len)),
                discard_incomplete_paths=False
            )
            paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_self_sup_paths(paths)
            self.expl_data_collector.end_epoch(-1)

        #TODO: where is the skill set and changed?

        for epoch in tqdm(range(self._start_epoch, self.num_epochs)):

            for train_loop in range(self.num_train_loops_per_epoch):

                # Explore
                if train_loop % self.seq_len == 0:
                    self.expl_data_collector.collect_new_paths(
                        seq_len=self.seq_len,
                        num_seqs=1,
                        discard_incomplete_paths=False
                    )

                # Train
                self.training_mode(True)

                for _ in range(self.num_trains_per_expl_seq):
                    train_data = self.replay_buffer.random_batch(self.batch_size)

                    # Train Latent
                    self.mode_latent_trainer.train(
                        skills=ptu.from_numpy(train_data.mode),
                        obs_seq=ptu.from_numpy(train_data.obs),
                        action_seq=ptu.from_numpy(train_data.action),
                    )

                    # Train SAC
                    self.trainer.train(train_data)

                self.training_mode(False)

            new_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_self_sup_paths(new_expl_paths)

            self._end_epoch(epoch)

    def _end_epoch(self, epoch):
        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)
        self.mode_latent_trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def training_mode(self, mode):
        for net in self.trainer.networks.values():
            net.train(mode)

    def to(self, device):
        for net in self.trainer.networks.values():
            net.to(device)
