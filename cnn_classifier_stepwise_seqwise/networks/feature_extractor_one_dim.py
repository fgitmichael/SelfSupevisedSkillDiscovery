import torch
from torch import nn

from cnn_classifier_stepwise.base.cnn_feature_extractor_base import \
    CnnFeatureExtractorBase
from cnn_classifier_stepwise_seqwise.utils.my_weights_init_xavier import \
    weights_init_xavier


class CnnFeatureExtractorOneDim(CnnFeatureExtractorBase):

    def create_cnn(self, params):
        obs_dim = params['obs_dim']
        dropout = params['dropout']

        self._outfeatures = obs_dim
        return nn.Sequential(
            nn.Conv1d(
                in_channels=obs_dim,
                out_channels=obs_dim,
                kernel_size=5,
                stride=1,
                padding=2,
            ).apply(weights_init_xavier),
            nn.BatchNorm1d(obs_dim),
            nn.Dropout(dropout),
            #nn.LeakyReLU(0.2),
        )

    @property
    def output_feature_size(self):
        return self._outfeatures

    def reshape_output(self,
                       output: torch.Tensor):
        """
        Args:
            output              : (N, data_dim, S)
        Return:
            out                 : (N, S, data_dim)
        """
        seq_dim = -1
        data_dim = 1
        return output.transpose(seq_dim, data_dim)

    def check_input(self,
                    input: torch.Tensor,
                    ):
        super().check_input(input)
        seq_dim = 1
        data_dim = -1
        assert input.size(seq_dim) > input.size(data_dim)

    def reshape_for_cnn(self,
                        input: torch.Tensor):
        seq_dim = 1
        data_dim = -1
        return input.transpose(data_dim, seq_dim)



