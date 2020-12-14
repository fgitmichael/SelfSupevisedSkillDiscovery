import gtimer as gt
import copy
from typing import List

from diayn_original_tb.algo.algo_diayn_tb_own_fun \
    import DIAYNTorchOnlineRLAlgorithmOwnFun

from latent_with_splitseqs.memory.replay_buffer_for_latent import LatentReplayBuffer
from latent_with_splitseqs.data_collector.seq_collector_split import SeqCollectorSplitSeq

from diayn_seq_code_revised.base.data_collector_base import PathCollectorRevisedBase


class SeqwiseAlgoRevisedSplitSeqs(DIAYNTorchOnlineRLAlgorithmOwnFun):

    def __init__(self,
                 *args,
                 batch_size,
                 horizon_len,
                 exploration_data_collector: SeqCollectorSplitSeq,
                 batch_size_latent=None,
                 train_sac_classifier_with_equal_data=False,
                 mode_influence_plotting=False,
                 **kwargs):
        super(SeqwiseAlgoRevisedSplitSeqs, self).__init__(
            *args,
            exploration_data_collector=exploration_data_collector,
            batch_size=batch_size,
            **kwargs
        )
        self.horizon_len = horizon_len
        self.mode_influence_plotting = mode_influence_plotting

        self.batch_size_latent = batch_size_latent \
            if batch_size_latent is not None \
            else self.batch_size

        if train_sac_classifier_with_equal_data is True and \
                batch_size_latent is not None:
            assert batch_size_latent == batch_size
        self.train_sac_classifier_with_equal_data = train_sac_classifier_with_equal_data

    def set_next_skill(self, data_collector: PathCollectorRevisedBase):
        data_collector.skill_reset()

    def _train(self):
        self.training_mode(False)

        if self.min_num_steps_before_training > 0:
            for _ in range(max(self.min_num_steps_before_training//self.horizon_len, 1)):
                self.set_next_skill(self.expl_data_collector)
                self._explore()

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
                    for train in range(num_trains_per_expl_step):
                        self._train_sac_latent()

            self._store_expl_data()
            self._end_epoch(epoch)

    def _explore(self):
        self.set_next_skill(self.expl_data_collector)
        self.expl_data_collector.collect_new_paths(
            seq_len=self.seq_len,
            num_seqs=1,
            horizon_len=self.horizon_len,
        )
        gt.stamp('exploration sampling', unique=False)

    def _sample_batch_for_latent_training_from_buffer(self):
        assert isinstance(self.replay_buffer, LatentReplayBuffer)
        train_data = self.replay_buffer.random_batch_latent_training(
            self.batch_size_latent
        )

        batch_dim = 0
        data_dim = 1
        seq_dim = 2
        for key, el in train_data.items():
            train_data[key] = el.transpose(batch_dim, seq_dim, data_dim)

        return train_data

    def _sample_batch_from_buffer(self) -> dict:
        train_data = super(SeqwiseAlgoRevisedSplitSeqs, self)._sample_batch_from_buffer()

        train_dict = dict(
            rewards=train_data.reward,
            terminals=train_data.terminal,
            observations=train_data.obs,
            actions=train_data.action,
            next_observations=train_data.next_obs,
            skills=train_data.mode,
        )

        return train_dict

    def _train_sac(self):
        raise NotImplementedError('Now not only sac is trained, buf also a latent model')

    def _train_sac_latent(self):
        self.training_mode(True)

        # Sample batch for sac training
        train_data_sac = self._sample_batch_from_buffer()

        # Sample batch for latent training
        train_data_latent = copy.deepcopy(train_data_sac) \
            if self.train_sac_classifier_with_equal_data \
            else self._sample_batch_for_latent_training_from_buffer()

        train_dict = dict(
            latent=train_data_latent,
            sac=train_data_sac,
        )
        self.trainer.train(train_dict)

        gt.stamp('training', unique=False)
        self.training_mode(False)

    def write_mode_influence_and_log(self, epoch):
        if self.mode_influence_plotting:
            super(SeqwiseAlgoRevisedSplitSeqs, self).write_mode_influence_and_log(epoch)
