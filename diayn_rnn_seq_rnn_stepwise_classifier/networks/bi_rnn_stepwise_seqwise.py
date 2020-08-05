import torch
from torch import nn

from code_slac.network.base import BaseNetwork
from self_supervised.network.flatten_mlp import FlattenMlp

from diayn_rnn_seq_rnn_stepwise_classifier.networks.positional_encoder import \
    PositionalEncoding
from diayn_rnn_seq_rnn_stepwise_classifier.networks.pos_encoder_oh import \
    PositionalEncodingOh
from diayn_rnn_seq_rnn_stepwise_classifier.networks.bi_rnn_stepwise import \
    BiRnnStepwiseClassifier

import self_supervised.utils.my_pytorch_util as my_ptu


class BiRnnStepwiseSeqWiseClassifier(BiRnnStepwiseClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size_rnn = kwargs['hidden_size_rnn']
        output_size = kwargs['output_size']
        hidden_sizes = kwargs['hidden_sizes']

        self.classifier_seq = FlattenMlp(
            input_size=self.rnn_params['num_features'],
            output_size=output_size,
            hidden_sizes=hidden_sizes
        )

    def forward(self,
                seq_batch,
                train=False):
        """
        Args:
            seq_batch           : (N, S, data_dim)
        Return:
            classified_steps    : (N, S, num_skills)
            classified_seqs     : (N, num_skills)

        """
        classified_steps, _, h_n = super().forward(
            seq_batch=seq_batch,
            return_rnn_outputs=True
        )

        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = seq_batch.size(batch_dim)
        seq_len = seq_batch.size(seq_dim)
        data_dim = seq_batch.size(data_dim)
        assert len(seq_batch.shape) == 3

        assert h_n.shape == torch.Size(
            (self.rnn_params['num_channels'],
             batch_size,
             self.rnn.hidden_size)
        )
        h_n = h_n.transpose(1, 0)
        assert my_ptu.tensor_equality(
            h_n.reshape(batch_size,
                        self.rnn_params['num_features'])[0],
            h_n[0].reshape(-1)
        )

        h_n = h_n.reshape(batch_size,
                          self.rnn_params['num_features'])

        classified_seqs = self.classifier_seq(h_n)
        assert classified_seqs.shape == torch.Size(
            (batch_size,
             self.classifier_seq.output_size)
        )

        if train:
            return classified_steps, classified_seqs

        else:
            return classified_steps
