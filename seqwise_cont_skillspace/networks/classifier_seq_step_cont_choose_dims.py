from typing import Union

from seqwise_cont_skillspace.networks.rnn_vae_classifier import \
    RnnVaeClassifierContSkills


class RnnStepwiseSeqwiseClassifierObsDimSelect(RnnVaeClassifierContSkills):

    def __init__(self,
                 *args,
                 input_size,
                 obs_dims_selected: Union[tuple, list]=None,
                 **kwargs,
                 ):
        if obs_dims_selected is not None:
            num_obs_selected = len(obs_dims_selected)
            input_size = num_obs_selected

            # Sanity check
            if 'obs_dims_selected' in kwargs.keys():
                raise ValueError('Double dim selected does not make sense')

        self.obs_dims_selected = obs_dims_selected

        super().__init__(
            *args,
            input_size=input_size,
            **kwargs,
        )


    def _process_seq(self, seq_batch):
        if self.obs_dims_selected is not None:
            seq_batch = seq_batch[..., self.obs_dims_selected]
        return super()._process_seq(seq_batch)
