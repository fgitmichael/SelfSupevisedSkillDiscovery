import abc
import torch

from code_slac.network.base import BaseNetwork


class CnnStepwiseClassifierNetBase(BaseNetwork, metaclass=abc.ABCMeta):

    def __init__(self,
                 obs_dim,
                 cnn_params):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = self.create_cnn(cnn_params)

    @abc.abstractmethod
    def create_cnn(self, params):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_feature_size(self):
        raise NotImplementedError

    def check_output(self,
                     output: torch.Tensor,
                     batch_size=None,
                     seq_len=None,
                     ):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        assert len(output.shape) == 3

        if batch_size is not None:
            assert output.size(batch_dim) == batch_size

        if seq_len is not None:
            assert output.size(seq_dim) == seq_len

        assert output.size(data_dim) == self.output_feature_size

    def check_input(self,
                    input: torch.Tensor,
                    ):
        """
        Args:
            input       : (N, S, data_dim)
        """
        assert len(input.shape) == 3

    def reshape_for_cnn(self,
                        input: torch.Tensor):
        """
        Args:
            input       : (N, S, data_dim)
        Return:
            reshaped    : (N, 1, S, data_dim)
        """
        return input.unsqueeze(dim=1)

    @abc.abstractmethod
    def reshape_output(self,
                       input: torch.Tensor):
        """
        Args:
            input           : tensor
        Return:
            output          : (N, S, feature_dim)
        """
        raise NotImplementedError

    def forward(self,
                seq: torch.Tensor,
                return_features_raw=False,
                ):
        batch_size, seq_len, data_dim = seq.shape
        self.check_input(seq)
        seq_for_cnn = self.reshape_for_cnn(seq)
        features_raw = self.net(seq_for_cnn)

        feature_seq = self.reshape_output(features_raw)
        self.check_output(feature_seq,
                          batch_size=batch_size,
                          seq_len=seq_len,
                          )

        if return_features_raw:
            return feature_seq, features_raw

        else:
            return feature_seq
