import numpy as np
from typing import Type

from latent_with_splitseqs.memory.replay_buffer_for_latent import LatentReplayBuffer


#class LatentReplayBufferSplitSeqSamplingRandomSeqLen(LatentReplayBufferSplitSeqSamplingBase):
#
#    def __init__(self,
#                 *args,
#                 min_sample_seq_len,
#                 max_sample_seq_len,
#                 **kwargs
#                 ):
#        super().__init__(
#            *args,
#            **kwargs
#        )
#        self.sample_seq_len_dict = dict(
#            low=min_sample_seq_len,
#            high=max_sample_seq_len,
#        )
#
#def _get_sample_seqlen(self) -> int:
#    return np.random.randint(**self.sample_seq_len_dict)
#
#def _add_sample_if(self, path_len) -> bool:
#    return not path_len < self.sample_seq_len_dict['low']


def get_random_seqlen_latent_replay_buffer_class(
        latent_replay_buffer_splitseq_cls: Type[LatentReplayBuffer],
) -> Type[LatentReplayBuffer]:

    class LatentReplayBufferRandomSeqLen(latent_replay_buffer_splitseq_cls):
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

    return LatentReplayBufferRandomSeqLen
