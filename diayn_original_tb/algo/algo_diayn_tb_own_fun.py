import torch
import gtimer as gt


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


class DIAYNTorchOnlineRLAlgorithmOwnFun(BaseRLAlgorithm):

    def __init__(self,
                 trainer: DIAYNTrainer,
                 exploration_env: NormalizedBoxEnvWrapper,
                 evaluation_env: NormalizedBoxEnvWrapper,
                 exploration_data_collector: SeqCollector,
                 evaluation_data_collector: SeqCollector,
                 replay_buffer: SelfSupervisedEnvSequenceReplayBufferDiscreteSkills,

                 seq_len,

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
        )
        self.seq_len = seq_len

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

        assert self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop, \
            'Online training presumes ' \
            'num_trains_per_train_loop >= num_expl_steps_per_train_loop'

        # get policy object for assigning skill
        self.policy = trainer.policy

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def _train(self):
        self.training_mode(False)
        if self.min_num_steps_before_training > 0:
            self.expl_data_collector.collect_new_paths(
                seq_len=self.seq_len,
                num_seqs=1,
                discard_incomplete_paths=False
            )
            init_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            gt.stamp('initial exploration', unique=True)

            for epoch in gt.timed_for(range(self._start_epoch, self.num_epochs)):
                # set policy for one epoch
                self.policy.skill_reset()

                num_trains_per_expl_step = self.num_train_loops_per_epoch // \
                    (self.num_expl_steps_per_train_loop * self.seq_len)
                num_trains_per_expl_step = min(num_trains_per_expl_step, 1)
                for _ in range(self.num_train_loops_per_epoch):
                    for _ in range(self.num_expl_steps_per_train_loop):
                        self.expl_data_collector.collect_new_steps(
                            seq_len=self.seq_len,
                            num_seqs=1,
                            discard_incomplete_paths=False
                        )
                        gt.stamp('exploration sampling', unique=False)

                        self.training_mode(True)

                        for _ in range(num_trains_per_expl_step):
                            train_data = self.replay_buffer.random_batch(
                                self.batch_size
                            )
                            self.trainer.train(train_data)
                        gt.stamp('training', unique=False)
                        self.training_mode(False)

                new_expl_paths = self.expl_data_collector.get_epoch_paths()
                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self._end_epoch(epoch)
