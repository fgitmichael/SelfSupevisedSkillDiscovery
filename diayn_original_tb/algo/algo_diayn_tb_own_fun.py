import torch
import gtimer as gt
import random
from typing import Union


import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.diayn.diayn_env_replay_buffer import DIAYNEnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.diayn.diayn_path_collector import DIAYNMdpPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.sac.diayn.diayn import DIAYNTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.diayn.diayn_torch_online_rl_algorithm import \
    DIAYNTorchOnlineRLAlgorithm
from rlkit.core.rl_algorithm import BaseRLAlgorithm

from self_supervised.base.writer.writer_base import WriterBase
from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.network.flatten_mlp import FlattenMlp as \
    MyFlattenMlp
from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb
from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter
from self_sup_comb_discrete_skills.data_collector.path_collector_discrete_skills import \
    PathCollectorSelfSupervisedDiscreteSkills
from self_sup_comb_discrete_skills.memory.replay_buffer_discrete_skills import \
    SelfSupervisedEnvSequenceReplayBufferDiscreteSkills

from diayn_original_tb.seq_path_collector.rkit_seq_path_collector import SeqCollector
from diayn_original_tb.policies.diayn_policy_extension import \
    SkillTanhGaussianPolicyExtension, MakeDeterministicExtension

from diayn_with_rnn_classifier.algo.diayn_trainer_with_rnn_classifier import \
    DIAYNTrainerRnnClassifierExtension


class DIAYNTorchOnlineRLAlgorithmOwnFun(DIAYNTorchOnlineRLAlgorithmTb):

    def __init__(self,
                 trainer: Union[
                          DIAYNTrainer,
                          DIAYNTrainerRnnClassifierExtension],

                 exploration_env: NormalizedBoxEnvWrapper,
                 evaluation_env: NormalizedBoxEnvWrapper,
                 exploration_data_collector: SeqCollector,
                 evaluation_data_collector: SeqCollector,
                 replay_buffer: SelfSupervisedEnvSequenceReplayBufferDiscreteSkills,

                 seq_len,
                 diagnostic_writer: DiagnosticsWriter,
                 seq_eval_collector: SeqCollector,

                 batch_size,
                 max_path_length,
                 num_epochs,
                 num_eval_steps_per_epoch,
                 num_expl_steps_per_train_loop,
                 num_trains_per_train_loop,
                 num_train_loops_per_epoch=1,
                 min_num_steps_before_training=0):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            diagnostic_writer=diagnostic_writer,
            seq_eval_collector=seq_eval_collector,
            batch_size=batch_size,
            max_path_length=max_path_length,
            num_epochs=num_epochs,
            num_eval_steps_per_epoch=num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop=num_expl_steps_per_train_loop,
            num_trains_per_train_loop=num_trains_per_train_loop,
            num_train_loops_per_epoch=num_train_loops_per_epoch,
            min_num_steps_before_training=min_num_steps_before_training,
        )

        self.seq_len = seq_len

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def set_next_skill(self, data_collector: SeqCollector):
        data_collector.set_skill(random.randint(0, self.policy.skill_dim - 1))

    def _train(self):
        self.training_mode(False)
        if self.min_num_steps_before_training > 0:
            self.set_next_skill(self.expl_data_collector)
            self.expl_data_collector.collect_new_paths(
                seq_len=self.seq_len,
                num_seqs=max(self.min_num_steps_before_training//self.seq_len, 1),
                discard_incomplete_paths=False
            )
            init_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_self_sup_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            gt.stamp('initial exploration', unique=True)

        num_trains_per_expl_step = self.num_trains_per_train_loop \
            // self.num_expl_steps_per_train_loop
        for epoch in gt.timed_for(range(
            self._start_epoch, self.num_epochs),
            save_itrs=True):

            self.set_next_skill(self.expl_data_collector)
            for train_loop in range(self.num_train_loops_per_epoch):
                for expl_step in range(self.num_expl_steps_per_train_loop):
                    self._explore()
                    for train in range(num_trains_per_expl_step * self.seq_len):
                        self._train_sac()

            self._store_expl_data()
            self._end_epoch(epoch)

    def _explore(self):
        self.set_next_skill(self.expl_data_collector)
        self.expl_data_collector.collect_new_paths(
            seq_len=self.seq_len,
            num_seqs=1,
            discard_incomplete_paths=False
        )
        gt.stamp('exploration sampling', unique=False)

    def _train_sac(self):
        self.training_mode(True)

        train_data = self.replay_buffer.random_batch(
            self.batch_size
        )

        batch_dim = 0
        data_dim = 1
        seq_dim = 2

        obs_dim = train_data.obs.shape[data_dim]
        action_dim = train_data.action.shape[data_dim]
        mode_dim = train_data.mode.shape[data_dim]

        train_data = train_data.transpose(batch_dim, seq_dim, data_dim)

        self.trainer.train(
            dict(
                rewards=train_data.reward.reshape(-1, 1),
                terminals=train_data.terminal.reshape(-1, 1),
                observations=train_data.obs.reshape(-1, obs_dim),
                actions=train_data.action.reshape(-1, action_dim),
                next_observations=train_data.next_obs.reshape(-1, obs_dim),
                skills=train_data.mode.reshape(-1, mode_dim)
            )
        )

        gt.stamp('training', unique=False)
        self.training_mode(False)

    def _store_expl_data(self):
        new_expl_paths = self.expl_data_collector.get_epoch_paths()
        self.replay_buffer.add_self_sup_paths(new_expl_paths)
        gt.stamp('data storing', unique=False)

