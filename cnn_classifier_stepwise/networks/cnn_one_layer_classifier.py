import torch
from torch import nn

from cnn_classifier_stepwise.base.cnn_feature_extractor_base import \
    CnnFeatureExtractorBase

from code_slac.network.base import weights_init_xavier


class CnnFeatureExtractorTwoDim(CnnFeatureExtractorBase):

    def create_cnn(self, params) -> nn.Module:
        channels = params['channels']
        dropout = params['dropout']

        self._out_features = channels[-1] * self.obs_dim
        return nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=channels[0],
                kernel_size=(51, 3),
                padding=(25, 1),
                stride=1
            ).apply(weights_init_xavier),
            nn.BatchNorm2d(channels[0]),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2),
            #nn.Conv2d(
            #    in_channels=channels[-2],
            #    out_channels=channels[-1],
            #    kernel_size=(5, 3),
            #    padding=(2, 1),
            #    stride=1
            #).apply(weights_init_xavier),
            #nn.BatchNorm2d(channels[-1]),
            #nn.Dropout(dropout),
        )

    @property
    def output_feature_size(self):
        return self._out_features

    def reshape_output(self,
                       input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input           : (N, out_channels, S, data_dim)
        Return:
            out             : (N, S, out_channels * data_dim)
        """
        batch_size, out_channels, seq_len, data_dim = input.shape
        in_transposed = input.permute(0, 2, 1, 3)
        in_reshaped = in_transposed.reshape(
            batch_size,
            seq_len,
            out_channels * data_dim
        )
        return in_reshaped
