import torch
from torch import nn

from code_slac.network.base import BaseNetwork
from self_supervised.network.flatten_mlp import FlattenMlp, FlattenMlpDropout

from diayn_rnn_seq_rnn_stepwise_classifier.networks.positional_encoder import \
    PositionalEncoding
from diayn_rnn_seq_rnn_stepwise_classifier.networks.pos_encoder_oh import \
    PositionalEncodingOh
from diayn_rnn_seq_rnn_stepwise_classifier.networks.bi_rnn_stepwise import \
    BiRnnStepwiseClassifier

import self_supervised.utils.my_pytorch_util as my_ptu


class BiRnnStepwiseSeqWiseClassifier(BiRnnStepwiseClassifier):

    def __init__(self,
                 *args,
                 dropout=0.,
                 layer_norm=False,
                 **kwargs):
        super().__init__(
            *args,
            dropout=dropout,
            layer_norm=layer_norm,
            **kwargs)
        self.output_size = kwargs['output_size']
        hidden_sizes = kwargs['hidden_sizes']

        self.classifier_seq = self.create_classifier_seq(
            input_size=self.rnn_params['num_features_h_n'],
            output_size=self.output_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def create_classifier_seq(self,
                              input_size,
                              output_size,
                              hidden_sizes,
                              dropout,
                              layer_norm=False,
                              ):
        return FlattenMlpDropout(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def _classify_stepwise(self, hidden_seq):
        """
        This method classifies STEP(!)-wise
        Detach stepwise classification from the rnn calculation as rnn should only
        be optimized based on the seq classification
        """
        out = super()._classify_stepwise(hidden_seq.detach())
        return out

    def _classify_seq_seqwise(self, h_n):
        """
        This method classfies SEQ(!)-wise
        """
        batch_size = h_n.size(1)

        assert h_n.shape == torch.Size(
            (self.rnn_params['num_channels'],
             batch_size,
             self.rnn.hidden_size)
        )
        h_n = h_n.transpose(1, 0)
        assert my_ptu.tensor_equality(
            h_n.reshape(batch_size,
                        self.rnn_params['num_features_h_n'])[0],
            h_n[0].reshape(-1)
        )

        h_n = h_n.reshape(batch_size,
                          self.rnn_params['num_features_h_n'])

        classified_seqs = self.classifier_seq(h_n)
        assert classified_seqs.shape == torch.Size(
            (batch_size,
             self.classifier_seq.output_size)
        )

        return classified_seqs

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
        assert len(seq_batch.shape) == 3

        classified_steps, _, h_n = super().forward(
            seq_batch=seq_batch,
            return_rnn_outputs=True
        )

        classified_seqs = self._classify_seq_seqwise(h_n)

        if train:
            return classified_steps, classified_seqs

        else:
            return classified_steps
