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
from rlkit.torch.sac.diayn.diayn_env_replay_buffer import DIAYNEnvReplayBuffer

from code_slac.network.latent import Gaussian


class DfMemoryEvalDIAYNCont(MemoryEvalBase):

    def __init__(
            self,
            *args,
            replay_buffer: DIAYNEnvReplayBuffer,
            df_to_evaluate: Gaussian,
            batch_size=64,
            **kwargs
    ):
        super().__init__(
            *args,
            replay_buffer=replay_buffer,
            df_to_evaluate=df_to_evaluate,
            **kwargs
        )
        self.batch_size = batch_size

    def sample_paths_from_replay_buffer(self):
        assert isinstance(self.replay_buffer, DIAYNEnvReplayBuffer)
        sampled_transitions = self.replay_buffer.random_batch(batch_size=self.batch_size)

        return dict(
            sampled_transitions=sampled_transitions,
        )

    @torch.no_grad()
    def apply_df(
            self,
            *args,
            sampled_transitions,
            **kwargs
    ) -> dict:
        next_obs = ptu.from_numpy(sampled_transitions['next_observations'])
        skill_recon_dist = my_ptu.eval(self.df_to_evaluate, next_obs)
        return dict(
            skill_recon_dist=skill_recon_dist
        )

    def classifier_evaluation(
            self,
            *args,
            epoch,
            skill_recon_dist,
            sampled_transitions,
            **kwargs
    ):
        df_accuracy_memory = F.mse_loss(
            skill_recon_dist.loc,
            ptu.from_numpy(sampled_transitions['skills'])
        )

        self.diagno_writer.writer.writer.add_scalar(
            tag=self.get_log_string("Classifier Performance/Memory"),
            scalar_value=df_accuracy_memory,
            global_step=epoch
        )
