import gtimer as gt
import random
from typing import Union
import numpy as np

from rlkit.torch.sac.diayn.diayn import DIAYNTrainer

from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer
import self_supervised.utils.typed_dicts as td

from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter
from self_sup_comb_discrete_skills.memory.replay_buffer_discrete_skills import \
    SelfSupervisedEnvSequenceReplayBufferDiscreteSkills

from diayn_original_tb.seq_path_collector.rkit_seq_path_collector import SeqCollector
from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb

from diayn_with_rnn_classifier.trainer.diayn_trainer_with_rnn_classifier import \
    DIAYNTrainerRnnClassifierExtension
from diayn_with_rnn_classifier.trainer.seq_wise_trainer_with_diayn_classifier_vote import \
    DIAYNTrainerMajorityVoteSeqClassifier
from diayn_with_rnn_classifier.trainer.seq_wise_trainer import DIAYNTrainerSeqWise

from diayn_rnn_seq_rnn_stepwise_classifier.trainer.diayn_step_wise_rnn_trainer import \
    DIAYNStepWiseRnnTrainer
from diayn_rnn_seq_rnn_stepwise_classifier.trainer.diayn_step_wise_and_seq_wise_trainer \
    import DIAYNStepWiseSeqWiseRnnTrainer

from diayn_seq_code_revised.data_collector.seq_collector_revised import SeqCollectorRevised
from diayn_seq_code_revised.data_collector.seq_collector_revised_discrete_skills import \
    SeqCollectorRevisedDiscreteSkills
from diayn_seq_code_revised.trainer.trainer_seqwise_stepwise_revised import \
    DIAYNAlgoStepwiseSeqwiseRevisedTrainer
from diayn_seq_code_revised.trainer.trainer_seqwise_stepwise_revised_noid import \
    DIAYNAlgoStepwiseSeqwiseRevisedNoidTrainer
from diayn_seq_code_revised.trainer.trainer_seqwise_stepwise_revised_obsdim_single \
    import DIAYNAlgoStepwiseSeqwiseRevisedObsDimSingleTrainer

from seqwise_cont_skillspace.trainer.cont_skillspace_seqwise_trainer import \
    ContSkillTrainerSeqwiseStepwise
from seqwise_cont_skillspace.trainer.discrete_skillspace_seqwise_stepwise_revised_trainer \
    import DiscreteSkillTrainerSeqwiseStepwise
from seqwise_cont_skillspace.trainer.trainer_ssvaestyle import SsvaestyleSkillTrainer
from seqwise_cont_skillspace.trainer.trainer_guided_ssvae_style import \
    GuidedSsvaestyleTrainer
from seqwise_cont_skillspace.trainer.cont_skillspace_nocont_steprepeat_trainer \
    import ContSkillTrainerSeqwiseStepwiseStepRepeatTrainer
from seqwise_cont_skillspace.trainer.trainer_single_dims_cont import \
    ContSkillTrainerSeqwiseStepwiseSingleDims

from sequence_stepwise_only.trainer.stepwise_only_trainer_cont import \
    StepwiseOnlyTrainerCont

from two_d_navigation_demo.trainer.trainer_stepwise_only_discrete import \
    StepwiseOnlyDiscreteTrainer

from cnn_classifier_stepwise_seqwise.trainer.cnn_stepwise_seqwise_trainer import \
    CnnStepwiseSeqwiseTrainer

from seqwise_cont_highdimusingvae.trainer.seqwise_cont_highdimusingvae_trainer import \
    ContSkillTrainerSeqwiseStepwiseHighdimusingvae

class DIAYNTorchOnlineRLAlgorithmOwnFun(DIAYNTorchOnlineRLAlgorithmTb):

    def __init__(self,
                 trainer: Union[
                     DIAYNTrainer,
                     DIAYNTrainerRnnClassifierExtension,
                     DIAYNTrainerMajorityVoteSeqClassifier],

                 exploration_env: NormalizedBoxEnvWrapper,
                 evaluation_env: NormalizedBoxEnvWrapper,
                 exploration_data_collector: Union[
                     SeqCollector,
                     SeqCollectorRevised,
                     SeqCollectorRevisedDiscreteSkills
                 ],
                 evaluation_data_collector: Union[
                     SeqCollector,
                     SeqCollectorRevised,
                     SeqCollectorRevisedDiscreteSkills
                 ],
                 replay_buffer: Union[
                     SelfSupervisedEnvSequenceReplayBufferDiscreteSkills,
                     SelfSupervisedEnvSequenceReplayBuffer],

                 seq_len,
                 diagnostic_writer: DiagnosticsWriter,
                 seq_eval_collector: Union[
                     SeqCollector,
                     SeqCollectorRevised,
                     SeqCollectorRevisedDiscreteSkills],


                 batch_size,
                 max_path_length,
                 num_epochs,
                 num_eval_steps_per_epoch,
                 num_expl_steps_per_train_loop,
                 num_trains_per_train_loop,
                 num_train_loops_per_epoch=1,
                 min_num_steps_before_training=0,
                 mode_influence_one_plot_scatter=False,
                 mode_influence_paths_obs_lim=None,
                 ):
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
            mode_influence_one_plot_scatter=mode_influence_one_plot_scatter,
            mode_influence_paths_obs_lim=mode_influence_paths_obs_lim,
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
            for _ in range(max(self.min_num_steps_before_training//self.seq_len, 1)):
                self.set_next_skill(self.expl_data_collector)
                self.expl_data_collector.collect_new_paths(
                    seq_len=self.seq_len,
                    num_seqs=1,
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

    def _sample_batch_from_buffer(self) -> td.TransitonModeMappingDiscreteSkills:
        train_data = self.replay_buffer.random_batch(
            self.batch_size
        )

        batch_dim = 0
        data_dim = 1
        seq_dim = 2

        train_data = train_data.transpose(batch_dim, seq_dim, data_dim)

        return train_data

    def _train_sac(self):
        self.training_mode(True)

        train_data = self._sample_batch_from_buffer()

        if type(self.trainer) in [DIAYNTrainerRnnClassifierExtension,
                                  DIAYNTrainerMajorityVoteSeqClassifier,
                                  DIAYNStepWiseSeqWiseRnnTrainer,
                                  DIAYNTrainerSeqWise,
                                  DIAYNAlgoStepwiseSeqwiseRevisedTrainer,
                                  DIAYNAlgoStepwiseSeqwiseRevisedNoidTrainer,
                                  ContSkillTrainerSeqwiseStepwise,
                                  DiscreteSkillTrainerSeqwiseStepwise,
                                  DIAYNStepWiseRnnTrainer,
                                  SsvaestyleSkillTrainer,
                                  GuidedSsvaestyleTrainer,
                                  StepwiseOnlyTrainerCont,
                                  StepwiseOnlyDiscreteTrainer,
                                  CnnStepwiseSeqwiseTrainer,
                                  ContSkillTrainerSeqwiseStepwiseStepRepeatTrainer,
                                  ContSkillTrainerSeqwiseStepwiseHighdimusingvae,
                                  DIAYNAlgoStepwiseSeqwiseRevisedObsDimSingleTrainer,
                                  ContSkillTrainerSeqwiseStepwiseSingleDims,
                                  ]:
            train_dict = dict(
                rewards=train_data.reward,
                terminals=train_data.terminal,
                observations=train_data.obs,
                actions=train_data.action,
                next_observations=train_data.next_obs,
                skills=train_data.mode,
            )
            if isinstance(self.replay_buffer,
                          SelfSupervisedEnvSequenceReplayBufferDiscreteSkills):
                train_dict['skills_id'] = train_data.skill_id

        else:
            data_dim = -1
            obs_dim = train_data.obs.shape[data_dim]
            action_dim = train_data.action.shape[data_dim]
            mode_dim = train_data.mode.shape[data_dim]

            train_dict = dict(
                rewards=train_data.reward.reshape(-1, 1),
                terminals=train_data.terminal.reshape(-1, 1),
                observations=train_data.obs.reshape(-1, obs_dim),
                actions=train_data.action.reshape(-1, action_dim),
                next_observations=train_data.next_obs.reshape(-1, obs_dim),
                skills=train_data.mode.reshape(-1, mode_dim),
            )

        self.trainer.train(train_dict)

        gt.stamp('training', unique=False)
        self.training_mode(False)

    def _store_expl_data(self):
        new_expl_paths = self.expl_data_collector.get_epoch_paths()
        self.replay_buffer.add_self_sup_paths(new_expl_paths)
        gt.stamp('data storing', unique=False)
