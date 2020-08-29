import torch
from torch import nn

from cnn_classifier_stepwise_seqwise.base.cnn_seqwise_classifier_base import \
    CnnForClassificationSeqwiseBase

from mode_disent_no_ssm.utils.empty_network import Empty

from self_supervised.network.flatten_mlp import FlattenMlpDropout


class CnnClassifierSeqwiseNoRawTwoDim(CnnForClassificationSeqwiseBase):

    def create_seqwise_classifier(self, params) -> FlattenMlpDropout:
        # num_channels * dim1 * dim2 (see reshape for classifier)
        num_in_features = params['num_in_features']
        hidden_sizes = params['hidden_sizes']
        dropout = params['dropout']
        return FlattenMlpDropout(
            input_size=num_in_features,
            output_size=self.num_skills,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

    def create_cnn_raw_processor(self, params) -> nn.Module:
        #in_channels = params['in_channels']
        #dropout = params['dropout']
        #return nn.Sequential(
        #    # (N, C, 100, 2) -> (N, C, 10, 2)
        #    nn.MaxPool2d(
        #        kernel_size=(10, 1),
        #    ),
        #    nn.BatchNorm2d(in_channels),
        #    nn.Dropout(dropout),
        #)
        return Empty()

    def reshape_processedraw_for_seqwise_classifier(self, input: torch.Tensor) \
            -> torch.Tensor:
        """
        Args:
            input           : (N, num_channels, dim1, dim2), dimensions depend on the raw
                              processor
        Return:
            out             : (N, num_channels * dim1 * dim2)
        """
        return input.reshape(input.size(0), -1)
