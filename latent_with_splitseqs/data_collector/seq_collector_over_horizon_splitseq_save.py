from latent_with_splitseqs.base.seq_collector_over_horizon_base import SeqCollectorHorizonBase


class SeqCollectorHorizonSplitSeqSaving(SeqCollectorHorizonBase):

    def _save_split_seq(self, *args, split_seq, **kwargs):
        self._epoch_split_seqs.append(split_seq)
