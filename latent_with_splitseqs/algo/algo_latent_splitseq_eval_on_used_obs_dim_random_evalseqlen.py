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

#    def _get_paths_mode_influence_test(self,
#                                       num_paths=1,
#                                       rollout_seqlengths_dict=None) \
#        -> List[td.TransitionModeMapping]:
#        assert isinstance(self.seq_eval_collector, SeqCollectorRevisedOptionalSkillId)
#
#        if rollout_seqlengths_dict is not None:
#            seq_len = rollout_seqlengths_dict['horizon_len']
#
#        else:
#            raise NotImplementedError
#
#        for skill_id, skill in enumerate(
#                self.seq_eval_collector.skill_selector.get_skill_grid()):
#
#            self.seq_eval_collector.skill = skill
#            self.seq_eval_collector.collect_new_paths(
#                seq_len=seq_len,
#                num_seqs=num_paths,
#                skill_id=skill_id # Use the Option to assign skill id
#            )
#
#        mode_influence_eval_paths = self.seq_eval_collector.get_epoch_paths()
#        assert type(mode_influence_eval_paths) is list
#        assert type(mode_influence_eval_paths[0]) \
#               is td.TransitonModeMappingDiscreteSkills
#        assert len(mode_influence_eval_paths) < self.seq_eval_collector.maxlen
#
#        return mode_influence_eval_paths

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
