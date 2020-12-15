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
            **kwargs
    ):
        assert isinstance(exploration_data_collector, SeqCollectorHorizonBase)
        super(SeqwiseAlgoSplitHorizonExplCollection, self).__init__(
            *args,
            exploration_data_collector=exploration_data_collector,
            **kwargs
        )

    def _initial_exploration(self):
        if self.min_num_steps_before_training > 0:
            self.set_next_skill(self.expl_data_collector)
            for _ in range(max(self.min_num_steps_before_training, 1)):
                self._explore()
        gt.stamp('initial exploration', unique=True)
        self._store_expl_data()
        self.expl_data_collector.end_epoch(-1)

    def _explore(self):
        assert isinstance(self.expl_data_collector, SeqCollectorHorizonBase)
        horizon_complete = self.expl_data_collector.collect_split_seq(
            seq_len=self.seq_len,
            horizon_len=self.horizon_len,
            discard_incomplete_seq=False,
        )
        if horizon_complete:
            self.set_next_skill(self.expl_data_collector)
        gt.stamp('exploration sampling', unique=False)
