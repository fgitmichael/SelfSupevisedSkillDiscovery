import gtimer as gt
import numpy as np
from typing import List

from latent_with_splitseqs.algo.algo_latent_splitseq_with_eval_on_used_obsdim \
    import SeqwiseAlgoRevisedSplitSeqsEvalOnUsedObsDim

import self_supervised.utils.typed_dicts as td

from seqwise_cont_skillspace.data_collector.seq_collector_optional_skill_id import \
    SeqCollectorRevisedOptionalSkillId


class SeqwiseAlgoRevisedSplitSeqsEvalOnUsedObsDimRandomSeqevalLen(
    SeqwiseAlgoRevisedSplitSeqsEvalOnUsedObsDim):

    def __init__(self,
                 *args,
                 min_seq_eval_len,
                 max_seq_eval_len,
                 **kwargs):
        super(SeqwiseAlgoRevisedSplitSeqsEvalOnUsedObsDimRandomSeqevalLen, self).__init__(
            *args,
            **kwargs
        )

        self.sample_seq_eval_len_dict = dict(
            low=min_seq_eval_len,
            high=max_seq_eval_len,
        )

    @property
    def seq_eval_len(self):
        return np.random.randint(**self.sample_seq_eval_len_dict)

    def _explore(self):
        self.set_next_skill(self.expl_data_collector)
        self.expl_data_collector.collect_new_paths(
            seq_len=self.seq_len,
            num_seqs=1,
            discard_incomplete_paths=False
        )
        gt.stamp('exploration sampling', unique=False)

    def classifier_perf_eval_log(self, epoch):
        # Get random seq_eval_len (each time different)
        seq_eval_len = self.seq_eval_len
        classifier_accuracy_eval_ret_dict = self.classifier_perf_eval(
            seq_eval_len=seq_eval_len
        )

        # From here on same as base method
        classifier_accuracy_eval = classifier_accuracy_eval_ret_dict['df_accuracy']
        post_figures = classifier_accuracy_eval_ret_dict['figs']
        self.diagnostic_writer.writer.writer.add_scalar(
            tag="Classifier Performance/Eval",
            scalar_value=classifier_accuracy_eval,
            global_step=epoch,
        )
        for key, fig in post_figures.items():
            self.diagnostic_writer.writer.writer.add_figure(
                tag="Rnn Debug/Mode Post Plot of evaluation "
                    "sequences from environment {}"
                    .format(key),
                figure=fig,
                global_step=epoch
            )

        classifier_accuracy_memory = self.classifier_perf_memory()
        self.diagnostic_writer.writer.writer.add_scalar(
            tag="Classifier Performance/Memory",
            scalar_value=classifier_accuracy_memory,
            global_step=epoch,
        )
