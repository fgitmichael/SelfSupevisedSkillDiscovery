import torch
from operator import itemgetter
from torch import nn
from torch.nn import functional as F
from self_supervised.network.flatten_mlp import FlattenMlp, FlattenMlpDropout

from diayn_rnn_seq_rnn_stepwise_classifier.networks.bi_rnn_stepwise_seqwise import \
    BiRnnStepwiseSeqWiseClassifier

from code_slac.network.base import BaseNetwork

from two_d_navigation_demo.base.create_posencoder import create_pos_encoder


class BiRnnStepwiseSeqWiseClassifierSingleDims(BaseNetwork):

    def __init__(self,
                 input_size,
                 hidden_size_rnn,
                 feature_size,
                 skill_dim,
                 hidden_sizes_classifier_seq,
                 hidden_sizes_classifier_step,
                 hidden_size_feature_dim_matcher,
                 seq_len,
                 obs_dims_used: tuple = None,
                 dropout=0.,
                 pos_encoder_variant='transformer',
                 num_layers=1,
                 bias=True,
                 layer_norm=False,
                 ):
        """
        Args:
            input_size        : dimension of state representation
            hidden_size_rnn   : dimension of hidden state in the rnn
        """
        super(BaseNetwork, self).__init__()

        self.skill_dim = skill_dim
        self.input_size = input_size
        self.obs_dims_used = [i for i in range(input_size)] \
            if obs_dims_used is None \
            else obs_dims_used
        self.rnn = nn.GRU(
            input_size=1,
            hidden_size=hidden_size_rnn,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.,
            bias=bias,
        )

        self.rnn_params = {}
        self.rnn_params['num_directions'] = \
            2 if self.rnn.bidirectional else 1
        self.rnn_params['num_channels'] = \
            self.rnn_params['num_directions'] * self.rnn.num_layers
        self.rnn_params['num_features_h_n'] = \
            self.rnn_params['num_channels'] * self.rnn.hidden_size
        self.rnn_params['num_features_hidden_seq'] = \
            self.rnn.hidden_size * self.rnn_params['num_directions']

        self.hidden_features_dim_matcher = self.create_hidden_features_dim_matcher(
            input_size=len(self.obs_dims_used) * self.rnn_params['num_features_hidden_seq'],
            output_size=feature_size,
            hidden_sizes=hidden_size_feature_dim_matcher,
        )

        pos_enc_ret_dict =  create_pos_encoder(
            feature_dim=feature_size,
            seq_len=seq_len,
            pos_encoder_variant=pos_encoder_variant,
        )
        self.pos_encoder, self.pos_enc_feature_size = itemgetter(
            'pos_encoder',
            'pos_encoded_feature_dim',
        )(pos_enc_ret_dict)

        self.classifier = self.create_step_classifier(
            input_size=self.pos_enc_feature_size,
            output_size=skill_dim,
            hidden_sizes=hidden_sizes_classifier_step,
            dropout=dropout,
            layer_norm=layer_norm,
        )

        self.classifier_seq = self.create_seq_classifier(
            input_size=len(self.obs_dims_used) * self.rnn_params['num_features_h_n'],
            output_size=skill_dim,
            hidden_sizes=hidden_sizes_classifier_seq,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def create_hidden_features_dim_matcher(self, input_size, output_size, hidden_sizes):
        return nn.Sequential(
            FlattenMlpDropout(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=hidden_sizes,
            ),
            nn.LogSoftmax(dim=-1),
        )

    def create_seq_classifier(self,
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
            layer_norm=layer_norm,
            dropout=dropout
        )

    def create_step_classifier(self,
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
            layer_norm=layer_norm,
            dropout=dropout
        )

    def _process_seq(self, seq_batch):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = seq_batch.size(batch_dim)
        seq_len = seq_batch.size(seq_dim)

        if self.obs_dims_used is not None:
            seq_batch = seq_batch[:, :, self.obs_dims_used]

        hidden_seqs = []
        h_n_s = []
        for dim in range(seq_batch.size(data_dim)):
            hidden_seq, h_n = self.rnn(seq_batch[..., [dim]])
            hidden_seqs.append(hidden_seq)
            h_n_s.append(h_n)

        assert hidden_seqs[0].shape == torch.Size(
            (batch_size,
             seq_len,
             self.rnn_params['num_features_hidden_seq'])
        )
        assert h_n_s[0].shape == torch.Size(
            (
                self.rnn_params['num_channels'],
                batch_size,
                self.rnn.hidden_size
            )
        )

        hidden_seqs = torch.cat(hidden_seqs, dim=data_dim)
        h_n_s = torch.cat(h_n_s, dim=data_dim)
        h_n_s = h_n_s.transpose(1, 0)
        h_n_s = h_n_s.reshape(
            batch_size,
            self.rnn_params['num_features_h_n'] * len(self.obs_dims_used),
        )

        return hidden_seqs, h_n_s

    @property
    def output_size(self):
        return self.skill_dim

    def forward(self, seq_batch, train=False):
        """
        Args:
            seq_batch           : (N, S, data_dim)
        Return:
            classified_steps    : (N, S, num_skills)
            classified_seqs     : (N, num_skills)
        """
        assert len(seq_batch.shape) == 3

        hidden_seqs, h_n_s = self._process_seq(seq_batch)
        hidden_seqs = hidden_seqs.detach()
        hidden_seqs_feature_matched = self.hidden_features_dim_matcher(hidden_seqs)
        hidden_seqs_feature_matched_pos_enc = self.pos_encoder(
            hidden_seqs_feature_matched)

        classified_steps = self.classifier(hidden_seqs_feature_matched_pos_enc)
        classified_seqs = self.classifier_seq(h_n_s)

        if train:
            return classified_steps, classified_seqs
        else:
            return classified_steps
