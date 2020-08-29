import torch
from torch import nn

from cnn_classifier_stepwise_seqwise.base.cnn_seqwise_classifier_base import \
    CnnForClassificationSeqwiseBase

from mode_disent_no_ssm.utils.empty_network import Empty

from self_supervised.network.flatten_mlp import FlattenMlpDropout


class CnnClassifierSeqwiseNoRawOneDim(CnnForClassificationSeqwiseBase):

    def create_seqwise_classifier(self, params) -> nn.Module:
        num_in_features = params['num_in_features']
        hidden_sizes = params['hidden_sizes']
        dropout = params['dropout']
        return FlattenMlpDropout(
            input_size=num_in_features,
            output_size=self.num_skills,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

    def reshape_processedraw_for_seqwise_classifier(self, input: torch.Tensor) \
        -> torch.Tensor:
        return input.reshape(input.size(0), -1)

    def create_cnn_raw_processor(self, params) -> nn.Module:
        return Empty()

    def check_input(self,
                    features_raw: torch.Tensor
                    ):
        seq_dim = -1
        data_dim = 1
        assert len(features_raw.shape) == 3
        assert features_raw.size(seq_dim) > features_raw.size(data_dim)



