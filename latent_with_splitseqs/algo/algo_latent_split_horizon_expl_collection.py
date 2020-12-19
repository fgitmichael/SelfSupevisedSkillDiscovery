import gtimer as gt

from latent_with_splitseqs.algo.algo_latent_splitseqs import SeqwiseAlgoRevisedSplitSeqs

from latent_with_splitseqs.data_collector.seq_collector_over_horizon_splitseq_save \
    import SeqCollectorHorizonSplitSeqSaving
from latent_with_splitseqs.base.seq_collector_over_horizon_base import SeqCollectorHorizonBase

class SeqwiseAlgoSplitHorizonExplCollection(SeqwiseAlgoRevisedSplitSeqs):

    def __init__(
            self,
            *args,
            exploration_data_collector: SeqCollectorHorizonSplitSeqSaving,
            train_while_exploration=True,
            **kwargs
    ):
        assert isinstance(exploration_data_collector, SeqCollectorHorizonBase)
        super(SeqwiseAlgoSplitHorizonExplCollection, self).__init__(
            *args,
            exploration_data_collector=exploration_data_collector,
            **kwargs
        )
        self.train_while_exploration = train_while_exploration

    def _initial_exploration(self):
        if self.min_num_steps_before_training > 0:
            self.set_next_skill(self.expl_data_collector)
            for _ in range(max(self.min_num_steps_before_training, 1)):
                self._explore()
            gt.stamp('initial exploration', unique=True)
            self._store_expl_data()
        self.expl_data_collector.end_epoch(-1)

    def _train_loop(self):
        if self.train_while_exploration is True:
            super()._train_loop()

        elif self.train_while_exploration == 'train_expl_split':
            for epoch in gt.timed_for(range(
                    self._start_epoch, self.num_epochs),
                    save_itrs=True):

                self.set_next_skill(self.expl_data_collector)
                for train_loop in range(self.num_train_loops_per_epoch):

                    # Exploration
                    for expl_step in range(self.num_expl_steps_per_train_loop):
                        self._explore()
                    self._store_expl_data()

                    # Training
                    for train_step in range(self.num_trains_per_train_loop):
                        self._train_sac_latent()

                    self._end_epoch(epoch)

        elif self.train_while_exploration == 'train_while_expl_horizon_completed':
            hparam_adjust = self.horizon_len // self.seq_len
            num_trains_per_expl_step = self.num_trains_per_train_loop \
                                       // self.num_expl_steps_per_train_loop
            num_trains_per_expl_step *= hparam_adjust

            for epoch in gt.timed_for(range(
                    self._start_epoch, self.num_epochs),
                    save_itrs=True):

                self.set_next_skill(self.expl_data_collector)
                for train_loop in range(self.num_train_loops_per_epoch):
                    for expl_step in range(self.num_expl_steps_per_train_loop):
                        horizon_complete = self._explore()
                        if horizon_complete:
                            for train in range(num_trains_per_expl_step):
                                self._train_sac_latent()

                self._store_expl_data()
                self._end_epoch(epoch)

        else:
            raise NotImplementedError

    def _explore(self) -> bool:
        assert isinstance(self.expl_data_collector, SeqCollectorHorizonBase)
        horizon_complete = self.expl_data_collector.collect_split_seq(
            seq_len=self.seq_len,
            horizon_len=self.horizon_len,
            discard_incomplete_seq=False,
        )
        if horizon_complete:
            self.set_next_skill(self.expl_data_collector)
        gt.stamp('exploration sampling', unique=False)

        return horizon_complete
