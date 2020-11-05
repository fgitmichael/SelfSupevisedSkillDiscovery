import torch
import numpy as np
import torch.nn.functional as F

from latent_with_splitseqs.base.df_memory_eval_base \
    import MemoryEvalBase
from latent_with_splitseqs.base.classifier_base import SplitSeqClassifierBase
from latent_with_splitseqs.algo.post_epoch_func_gtstamp_wrapper \
    import post_epoch_func_wrapper

from self_supervised.memory.self_sup_replay_buffer \
    import SelfSupervisedEnvSequenceReplayBuffer
import self_supervised.utils.my_pytorch_util as my_ptu

import self_supervised.utils.typed_dicts as td

import rlkit.torch.pytorch_util as ptu

from self_sup_combined.base.writer.is_log import is_log


class DfMemoryEvalSplitSeq(MemoryEvalBase):

    def __init__(self,
                 *args,
                 seq_eval_len: int,
                 horizon_eval_len: int,
                 replay_buffer: SelfSupervisedEnvSequenceReplayBuffer,
                 df_to_evaluate: SplitSeqClassifierBase,
                 batch_size = 64,
                 **kwargs
                 ):
        super(DfMemoryEvalSplitSeq, self).__init__(
            *args,
            replay_buffer=replay_buffer,
            df_to_evaluate=df_to_evaluate,
            **kwargs
        )
        self.seq_eval_len = seq_eval_len
        self.horizon_eval_len = horizon_eval_len
        self.batch_size = batch_size

    @is_log()
    @post_epoch_func_wrapper(gt_stamp_name="df evaluation memory")
    def __call__(self, *args, **kwargs):
        super(DfMemoryEvalSplitSeq, self).__call__(*args, **kwargs)

    def sample_paths_from_replay_buffer(self):
        memory_paths = self.replay_buffer.random_batch_bsd_format(self.batch_size)

        return dict(
            memory_paths=memory_paths
        )

    @torch.no_grad()
    def apply_df(self, memory_paths: td.TransitionModeMapping):
        next_obs = ptu.from_numpy(memory_paths.next_obs)
        df_ret_dict = my_ptu.eval(self.df_to_evaluate, obs_seq=next_obs)
        return df_ret_dict

    def calc_classifier_performance(
            self,
            *args,
            epoch,
            skill_recon_dist,
            memory_paths: td.TransitionModeMapping,
            **kwargs
    ):
        df_accuracy_memory = F.mse_loss(
            skill_recon_dist.loc,
            ptu.from_numpy(memory_paths.mode[:, 0, :])
        )

        self.diagno_writer.writer.writer.add_scalar(
            tag="Classifier Performance/Memory",
            scalar_value=df_accuracy_memory,
            global_step=epoch
        )


