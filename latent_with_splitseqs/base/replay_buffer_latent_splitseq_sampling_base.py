import abc
import numpy as np

from latent_with_splitseqs.memory.replay_buffer_for_latent import LatentReplayBuffer

import self_supervised.utils.typed_dicts as td

from my_utils.np_utils.take_per_row import take_per_row
from my_utils.np_utils.np_array_equality import np_array_equality


class LatentReplayBufferSplitSeqSamplingBase(LatentReplayBuffer,
                                             metaclass=abc.ABCMeta):
    def __init__(
            self,
            *args,
            min_sample_seqlen: int = 2,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._min_sample_seqlen = min_sample_seqlen

    @abc.abstractmethod
    def _get_sample_seqlen(self) -> int:
        raise NotImplementedError

    @property
    def horizon_len(self):
        return self._seq_len

    @staticmethod
    def _add_left_zero_padding(
            arr: np.ndarray,
            padding_len: int,
            padding_dim: int
    ) -> np.ndarray:
        size_padding = list(arr.shape)
        size_padding[padding_dim] = padding_len
        padding_arr = np.zeros(size_padding)
        padded = np.concatenate([padding_arr, arr], axis=padding_dim)
        return padded

    def _take_elements_from_batch(
            self,
            batch: dict,
            batch_size: int,
            sample_seq_len: int,
    ) -> dict:
        """
        Prepends zero padding and samples sequences of seq_len from batch
        Args:
            batch               : dict with (N, data_dim, S) numpy arrays
            batch_size          : N
            sample_seq_len      : length of sampling that are taken along dimension S
        Return:
            sampled_batch       : dict with (N, data_dim, sample_seq_len) numpy arrays
        """
        batch_dim = 0
        data_dim = 1
        seq_dim = 2
        skill_key = 'mode'
        next_obs_key = 'next_obs'
        horizon_len = batch[next_obs_key].shape[seq_dim]
        transition_mode_mapping_kwargs = {}
        start_idx = np.random.randint(
            low=0,
            high=self.horizon_len, # padding is needed
            size=(batch_size,)
        )

        if self._padding:
            for key, el in batch.items():
                if isinstance(el, np.ndarray) and key != skill_key:
                    el_padded = self._add_left_zero_padding(
                        arr=el,
                        padding_len=sample_seq_len,
                        padding_dim=seq_dim,
                    )

                elif isinstance(el, np.ndarray) and key == skill_key:
                    assert np_array_equality(
                        el,
                        np.stack([el[..., 0]] * horizon_len, axis=seq_dim)
                    )
                    padding_arr = np.stack([el[..., 0]] * sample_seq_len, axis=seq_dim)
                    el_padded = np.concatenate([padding_arr, el], axis=seq_dim)

                else:
                    el_padded = None

                # Sample from seqs
                if isinstance(el_padded, np.ndarray):
                    el_bsd = np.swapaxes(el_padded, axis1=data_dim, axis2=seq_dim)
                    el_slices_bsd = take_per_row(el_bsd, start_idx, num_elem=sample_seq_len)
                    transition_mode_mapping_kwargs[key] = \
                        np.swapaxes(el_slices_bsd, axis1=1, axis2=2)

        else:
            end_idx_bool = start_idx > self.horizon_len - sample_seq_len
            start_idx[end_idx_bool] = np.random.randint(
                self.horizon_len - sample_seq_len,
                size=(batch_size,)
            )

            for key, el in batch.items():
                if isinstance(el, np.ndarray):
                    el_bsd = np.swapaxes(el, axis1=data_dim, axis2=seq_dim)
                    el_slices_bsd = take_per_row(el_bsd, start_idx, num_elem=sample_seq_len)
                    transition_mode_mapping_kwargs[key] \
                        = np.swapaxes(el_slices_bsd, axis1=1, axis2=2)

        return transition_mode_mapping_kwargs

    def random_batch(self,
                     batch_size: int) -> td.TransitionModeMapping:
        """
        Sample batches with random sequence length
        Args:
            batch_size                 : N
        Return:
            TransitionModeMapping      : consisting of (N, data_dim, S) tensors
        """
        sample_seq_len = self._get_sample_seqlen()
        batch_horizon = super().random_batch(
            batch_size=batch_size,
        )
        transition_mode_mapping_kwargs = self._take_elements_from_batch(
            batch=batch_horizon,
            batch_size=batch_size,
            sample_seq_len=sample_seq_len,
        )

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
        sample_seq_len = self._get_sample_seqlen()
        batch_horizon = super().random_batch_latent_training(
            batch_size=batch_size,
        )
        return_dict = self._take_elements_from_batch(
            batch=batch_horizon,
            sample_seq_len=sample_seq_len,
            batch_size=batch_size,
        )

        return return_dict

    def _add_sample_if(self, path_len) -> bool:
        """
        No terminal handling in this class
        """
        return True
