import torch
from typing import List


from diayn_with_rnn_classifier.algo.seq_wise_algo_classfier_perf_logging import \
    SeqWiseAlgoClassfierPerfLogging

from diayn_seq_code_revised.data_collector.seq_collector_revised_discrete_skills import \
    SeqCollectorRevisedDiscreteSkills
from diayn_seq_code_revised.data_collector.seq_collector_revised import \
    SeqCollectorRevised

import self_supervised.utils.typed_dicts as td


class SeqwiseAlgoRevised(SeqWiseAlgoClassfierPerfLogging):

    def set_next_skill(self, data_collector: SeqCollectorRevised):
        data_collector.skill_reset()


class SeqwiseAlgoRevisedDiscreteSkills(SeqwiseAlgoRevised):

    def _get_paths_mode_influence_test(self,
                                       num_paths=1,
                                       seq_len=200) \
            -> List[td.TransitonModeMappingDiscreteSkills]:
        assert isinstance(self.seq_eval_collector, SeqCollectorRevisedDiscreteSkills)

        for id, skill in enumerate(
                self.seq_eval_collector.skill_selector.get_skill_grid()):
            self.seq_eval_collector.skill = dict(
                skill=skill,
                id=id
            )
            self.seq_eval_collector.collect_new_paths(
                seq_len=seq_len,
                num_seqs=num_paths
            )

        mode_influence_eval_paths = self.seq_eval_collector.get_epoch_paths()
        return mode_influence_eval_paths
