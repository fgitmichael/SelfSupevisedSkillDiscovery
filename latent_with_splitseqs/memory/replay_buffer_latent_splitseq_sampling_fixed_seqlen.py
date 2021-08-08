import numpy as np
from typing import Type

from latent_with_splitseqs.memory.replay_buffer_for_latent import LatentReplayBuffer


def get_fixed_seqlen_latent_replay_buffer_class(
        latent_replay_buffer_splitseq_cls: Type[LatentReplayBuffer],
) -> Type[LatentReplayBuffer]:

    class LatentReplayBufferSplitSeqSamplingRandomSeqLen(
            latent_replay_buffer_splitseq_cls):

        def __init__(self,
                     *args,
                     sample_seqlen,
                     **kwargs
                     ):
            super().__init__(
                *args,
                **kwargs
            )
            self.sample_seqlen = sample_seqlen

        def _get_sample_seqlen(self) -> int:
            return self.sample_seqlen

        def _add_sample_if(self, path_len) -> bool:
            return not path_len < self.sample_seqlen

    return LatentReplayBufferSplitSeqSamplingRandomSeqLen
