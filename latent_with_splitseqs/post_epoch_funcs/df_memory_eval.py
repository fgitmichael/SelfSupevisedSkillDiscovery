import torch
import numpy as np
import torch.nn.functional as F

from latent_with_splitseqs.base.df_memory_eval_base \
    import MemoryEvalBase
from latent_with_splitseqs.base.classifier_base import SplitSeqClassifierBase

from self_supervised.memory.self_sup_replay_buffer \
    import SelfSupervisedEnvSequenceReplayBuffer
import self_supervised.utils.my_pytorch_util as my_ptu

import self_supervised.utils.typed_dicts as td

import rlkit.torch.pytorch_util as ptu


class DfMemoryEvalSplitSeq(MemoryEvalBase):

    def __init__(self,
                 *args,
                 seq_len: int,
                 horizon_len: int,
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
        self.seq_eval_len = seq_len
        self.horizon_eval_len = horizon_len
        self.batch_size = batch_size

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

    def classifier_evaluation(
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
            tag=self.get_log_string("Classifier Performance/Memory"),
            scalar_value=df_accuracy_memory,
            global_step=epoch
        )
