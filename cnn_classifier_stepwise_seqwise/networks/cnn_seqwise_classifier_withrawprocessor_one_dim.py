import torch
from torch import nn

from cnn_classifier_stepwise_seqwise.networks.cnn_seqwise_classifier_noraw_two_dim import CnnClassifierSeqwiseNoRawTwoDim


class CnnClassifierSeqwiseWithRaw1d(CnnClassifierSeqwiseNoRawTwoDim):

    def check_input(self,
                    features_raw: torch.Tensor
                    ):
        batch_size, data, seq_len = features_raw.shape
        assert len(features_raw.shape) == 3

    def create_cnn_raw_processor(self, params) -> nn.Module:
        in_channels = params['data_dim']
        dropout = params['dropout']

        return nn.Sequential(
            nn.MaxPool1d(
                kernel_size=10,
            ),
            nn.BatchNorm1d(in_channels),
            nn.Dropout(dropout),
        )
