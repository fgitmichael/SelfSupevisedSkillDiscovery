import abc
import torch
from torch import nn

from code_slac.network.base import BaseNetwork


class CnnForClassificationSeqwiseBase(BaseNetwork, metaclass=abc.ABCMeta):

    def __init__(self,
                 num_skills,
                 raw_processor_params,
                 seqwise_classifier_params,
                 ):
        super().__init__()
        self.num_skills = num_skills

        self.raw_processor = self.create_cnn_raw_processor(raw_processor_params)
        self.seq_classifier = self.create_seqwise_classifier(seqwise_classifier_params)

    @abc.abstractmethod
    def create_seqwise_classifier(self, params) -> nn.Module:
        raise NotImplementedError
    
    @abc.abstractmethod
    def reshape_processedraw_for_seqwise_classifier(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def create_cnn_raw_processor(self, params) -> nn.Module:
        raise NotImplementedError

    def check_output(self,
                     output: torch.Tensor,
                     batch_size=None,
                     seq_len=None,
                     ):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        assert len(output.shape) == 2

        if batch_size is not None:
            assert output.size(batch_dim) == batch_size

        if seq_len is not None:
            assert output.size(seq_dim) == seq_len

        assert output.size(data_dim) == self.num_skills

    def check_input(self,
                    features_raw: torch.Tensor
                    ):
        """
        input has to be raw output from cnn stepwise classifier
        """
        assert len(features_raw.shape) == 4

    def forward(self,
                features_raw: torch.Tensor,
                ):
        self.check_input(features_raw)
        processed_features_raw = self.raw_processor(features_raw)

        prepared_processed = self.reshape_processedraw_for_seqwise_classifier(processed_features_raw)
        classified_seqs = self.seq_classifier(prepared_processed)

        return classified_seqs
