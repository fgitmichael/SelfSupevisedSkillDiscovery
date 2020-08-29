from torch import nn

from cnn_classifier_stepwise_seqwise.networks.cnn_seqwise_classifier_noraw_two_dim import \
    CnnClassifierSeqwiseNoRawTwoDim


class CnnClassifierSeqwiseWithRaw2d(CnnClassifierSeqwiseNoRawTwoDim):

    def create_cnn_raw_processor(self, params) -> nn.Module:
        in_channels = params['in_channels']
        dropout = params['dropout']
        return nn.Sequential(
            # (N, C, 100, 2) -> (N, C, 10, 2)
            nn.MaxPool2d(
                kernel_size=(10, 1),
            ),
            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(dropout),
        )
