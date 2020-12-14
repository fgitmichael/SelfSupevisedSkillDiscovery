import numpy as np

from latent_with_splitseqs.memory.replay_buffer_for_latent import LatentReplayBuffer

import self_supervised.utils.typed_dicts as td

from my_utils.np_utils.create_aranges import create_aranges
from my_utils.np_utils.take_per_row import take_per_row


class LatentReplayBufferSplitSeqSampling(LatentReplayBuffer):

    def __init__(self,
                 *args,
                 min_sample_seq_len,
                 max_sample_seq_len,
                 **kwargs
                 ):
        super(LatentReplayBufferSplitSeqSampling, self).__init__(
            *args,
            **kwargs
        )
        self.sample_seq_len_dict = dict(
            low=min_sample_seq_len,
            high=max_sample_seq_len,
        )

    @property
    def horizon_len(self):
        return self._seq_len

    def random_batch(self,
                     batch_size: int) -> td.TransitionModeMapping:
        """
        Sample batches with random sequence length
        Args:
            batch_size                 : N
        Return:
            TransitionModeMapping      : consisting of (N, data_dim, S) tensors
        """
        sample_seq_len = np.random.randint(**self.sample_seq_len_dict)
        batch_size = (batch_size * self.horizon_len) // sample_seq_len
        batch_horizon = super().random_batch(
            batch_size=batch_size,
        )

        start_idx = np.random.randint(
            low=0,
            high=self.horizon_len - sample_seq_len,
            size=(batch_size,)
        )

        batch_dim = 0
        data_dim = 1
        seq_dim = 2
        transition_mode_mapping_kwargs = {}
        for key, el in batch_horizon.items():
            if isinstance(el, np.ndarray):
                el_bsd = np.swapaxes(el, axis1=data_dim, axis2=seq_dim)
                el_slices_bsd = take_per_row(el_bsd, start_idx, num_elem=sample_seq_len)
                transition_mode_mapping_kwargs[key] = \
                    np.swapaxes(el_slices_bsd, axis1=1, axis2=2)

        return td.TransitionModeMapping(
            **transition_mode_mapping_kwargs
        )

    def random_batch_latent_training(self,
                                     batch_size: int) -> dict:
        """
        Sample batches with random sequence length
        Args:
            batch_size              : N
        """
        sample_seq_len = np.random.randint(**self.sample_seq_len_dict)
        batch_size_adjusted = batch_size * self.horizon_len//sample_seq_len
        batch_horizon = super(LatentReplayBufferSplitSeqSampling, self).\
            random_batch_latent_training(
            batch_size=batch_size_adjusted,
        )

        start_idx = np.random.randint(
            low=0,
            high=self.horizon_len - sample_seq_len,
            size=(batch_size_adjusted,)
        )

        batch_dim = 0
        data_dim = 1
        seq_dim = 2
        return_dict = {}
        for key, el in batch_horizon.items():
            if isinstance(el, np.ndarray):
                el_bsd = np.swapaxes(el, axis1=data_dim, axis2=seq_dim)
                el_slices_bsd = take_per_row(el_bsd, start_idx, num_elem=sample_seq_len)
                return_dict[key] = \
                    np.swapaxes(el_slices_bsd, axis1=1, axis2=2)

        return return_dict
