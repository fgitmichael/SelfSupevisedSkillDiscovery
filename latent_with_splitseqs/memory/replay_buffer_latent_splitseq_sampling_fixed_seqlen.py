import numpy as np

from latent_with_splitseqs.base.replay_buffer_latent_splitseq_sampling_base \
    import LatentReplayBufferSplitSeqSamplingBase


class LatentReplayBufferSplitSeqSamplingRandomSeqLen(LatentReplayBufferSplitSeqSamplingBase):

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
