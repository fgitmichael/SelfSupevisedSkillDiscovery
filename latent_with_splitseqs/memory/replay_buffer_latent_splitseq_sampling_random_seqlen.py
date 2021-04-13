import numpy as np

from latent_with_splitseqs.base.replay_buffer_latent_splitseq_sampling_base \
    import LatentReplayBufferSplitSeqSamplingBase


class LatentReplayBufferSplitSeqSamplingRandomSeqLen(LatentReplayBufferSplitSeqSamplingBase):

    def __init__(self,
                 *args,
                 min_sample_seq_len,
                 max_sample_seq_len,
                 **kwargs
                 ):
        super().__init__(
            *args,
            **kwargs
        )
        self.sample_seq_len_dict = dict(
            low=min_sample_seq_len,
            high=max_sample_seq_len,
        )

    def _get_sample_seqlen(self) -> int:
        return np.random.randint(**self.sample_seq_len_dict)

    def _add_sample_if(self, path_len) -> bool:
        return not path_len < self.sample_seq_len_dict['low']
