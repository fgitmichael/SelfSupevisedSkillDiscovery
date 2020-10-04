from typing import List

import self_supervised.utils.typed_dicts as td

from latent_with_splitseqs.algo.algo_latent_splitseqs \
    import SeqwiseAlgoRevisedSplitSeqs
from latent_with_splitseqs.data_collector.seq_collector_split import SeqCollectorSplitSeq

from seqwise_cont_skillspace.algo.algo_cont_skillspace import  SeqwiseAlgoRevisedContSkills


class SeqwiseAlgoRevisedSplitSeqsEval(SeqwiseAlgoRevisedSplitSeqs):

    def __init__(self,
                 *args,
                 mode_influence_plotting=True,
                 seq_eval_len=200,
                 **kwargs,
                 ):
        super(SeqwiseAlgoRevisedSplitSeqsEval, self).__init__(
            *args,
            mode_influence_plotting=mode_influence_plotting,
            **kwargs
        )
        self.seq_eval_len = seq_eval_len

    def _get_paths_mode_influence_test(self, num_paths=1, seq_len=200) \
            -> List[td.TransitionModeMapping]:
        assert isinstance(self.seq_eval_collector, SeqCollectorSplitSeq)

        for skill_id, skill in enumerate(
                self.seq_eval_collector.skill_selector.get_skill_grid()):

            self.seq_eval_collector.skill = skill
            self.seq_eval_collector.collect_new_paths(
                seq_len=seq_len,
                num_seqs=num_paths,
                skill_id=skill_id # Use the Option to assign skill id
                # (new in SeqCollectorRevisedOptionalKSkillid)
            )

        mode_influence_eval_paths = self.seq_eval_collector.get_epoch_paths()
        assert type(mode_influence_eval_paths) is list
        assert type(mode_influence_eval_paths[0]) \
               is td.TransitonModeMappingDiscreteSkills

        return mode_influence_eval_paths


    def _log_net_param_hist(self, epoch):
        for k, net in self.trainer.network_dict.items():
            for name, weight in net.named_parameters():
                self.diagnostic_writer.writer.writer. \
                    add_histogram(k + name, weight, epoch)
                if weight.grad is not None:
                    self.diagnostic_writer.writer.writer. \
                        add_histogram(f'{k + name}.grad', weight.grad, epoch)

    def write_mode_influence_and_log(self, epoch):
        paths = self._get_paths_mode_influence_test(
            seq_len=self.seq_eval_len,
        )
        self._write_mode_influence_and_log(
            paths=paths,
            epoch=epoch,
        )

#    def _write_mode_influence(self,
#                              path,
#                              obs_dim,
#                              action_dim,
#                              epoch,
#                              obs_lim=None,
#                              ):
#        if self.trainer.df.obs_dims_used is None:
#            path.obs = path.obs[:self.trainer.df.obs_dimensions_used]
#        else:
#            path.obs = path.obs[list(self.trainer.df.obs_dims_used)]
#        if self.mode_influence_plotting:
#            super()._write_mode_influence(
#                path=path,
#                obs_dim=self.trainer.df.obs_dimensions_used \
#                    if self.trainer.df.obs_dims_used is None \
#                    else len(self.trainer.df.obs_dims_used),
#                action_dim=action_dim,
#                epoch=epoch
#            )
#