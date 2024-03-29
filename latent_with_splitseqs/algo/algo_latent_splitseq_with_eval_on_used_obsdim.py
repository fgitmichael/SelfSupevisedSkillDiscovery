from typing import Union

from latent_with_splitseqs.algo.algo_latent_splitseqs_with_eval \
    import SeqwiseAlgoRevisedSplitSeqsEval

import self_supervised.utils.typed_dicts as td


class SeqwiseAlgoRevisedSplitSeqsEvalOnUsedObsDim(SeqwiseAlgoRevisedSplitSeqsEval):

    def __init__(self,
                 *args,
                 obs_dims_to_log: Union[list, tuple] = None,
                 **kwargs):
        super(SeqwiseAlgoRevisedSplitSeqsEvalOnUsedObsDim, self).__init__(
            *args,
            **kwargs
        )

        if obs_dims_to_log is not None:
            self.obs_dims_to_log = obs_dims_to_log
        else:
            self.obs_dims_to_log = self.trainer.df.used_dims

        obs_dim = self.expl_env.observation_space.shape[0]
        self.obs_dims_real = [i for i in range(obs_dim)]

        # Sanity check
        assert all([True
                    if i in self.obs_dims_real
                    else False
                    for i in self.obs_dims_to_log])

    def write_mode_influence_and_log(self, epoch):
        if len(self.obs_dims_to_log) < 5:
            paths = self._get_paths_mode_influence_test(
                rollout_seqlengths_dict=dict(
                    seq_len=self.horizon_eval_len,
                    horizon_len=self.horizon_eval_len,
                )
            )

            # Extract relevant dimensions
            paths_relevant_dimensions = []
            seq_dim = 1
            data_dim = 0
            for path in paths:
                path_relevant_dimensions = td.TransitionModeMapping(
                    **path
                )
                path_relevant_dimensions.obs = \
                    path.obs[self.obs_dims_to_log, :]
                path_relevant_dimensions.next_obs = \
                    path.next_obs[ self.obs_dims_to_log]
                paths_relevant_dimensions.append(path_relevant_dimensions)

            self._write_mode_influence_and_log(
                paths=paths_relevant_dimensions,
                epoch=epoch,
            )
