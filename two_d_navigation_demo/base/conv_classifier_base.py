import torch
from torch import nn
from torch.nn import functional as F
import abc

from code_slac.network.base import BaseNetwork

from self_supervised.network.flatten_mlp import FlattenMlp
import self_supervised.utils.my_pytorch_util as my_ptu

from two_d_navigation_demo.base.seqwise_stepwise_classfier_base import \
    create_pos_encoder

from code_slac.network.base import weights_init_xavier

from self_supervised.network.flatten_mlp import FlattenMlpDropout

class StepwiseSeqwiseCnnEncoder(BaseNetwork):

    def __init__(self,
                 obs_dim,
                 num_skills,
                 hidden_sizes,
                 dropout,
                 seq_len,
                 out_channels,
    ):
        self.num_skills = num_skills

        super().__init__()
        # Inputs will be (N, 1, S, D)
        leaky_slope = 0.2
        kernel_size1 = (10, 10)
        kernel_size1 = (min(kernel_size1[0], seq_len),
                        min(kernel_size1[1], obs_dim))

        if kernel_size1[0] % 2 == 0:
            kernel_size1[0] += 1

        if kernel_size1[1] % 2 == 0:
            kernel_size1[1] += 1

        padding1 = (5, 1)
        stide = 1

        kernel_size2 = (3, 3)
        padding2 = (1, 1)

        self.feature_extractor = nn.Sequential(
            # (1, 100, 2) -> (3, 100, 2)
            nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=kernel_size1,
                padding=padding1,
                stride=1
            ),
            nn.LeakyReLU(leaky_slope),
            # (3, 100, 2) -> (32, 100, 2)
            nn.Conv2d(
                in_channels=3,
                out_channels=out_channels,
                kernel_size=kernel_size2,
                padding=padding2,
                stride=1
            ),
            nn.LeakyReLU
        ).apply(weights_init_xavier)

        # (32, 100, 2) -> (32, 50, 2)
        self.seq_wise_classifier_net1 = \
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=(1, 1),
                stride=(2, 1),
            )

        self.seq_wise_classifier_net2 = FlattenMlpDropout(
            input_size=32*50*2,
            output_size=num_skills,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

        self.stepwise_classifier_net = FlattenMlpDropout(
            input_size=32*2,
            output_size=num_skills,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

    def forward(self, seq: torch.Tensor):
        """
        Args:
            seq             : (N, S, obs_dim)
        """
        batch_dim = 0
        batch_size = seq.size(batch_dim)
        seq = seq.unsqueeze(dim=1)
        feature_seq = self.feature_extractor(seq)
        assert feature_seq.shape == torch.Size((batch_size, 32, 100, 2))

        feature_seq_reduced = self.seq_wise_classifier_net1(feature_seq)
        assert feature_seq_reduced.shape == torch.Size((batch_size, 32, 50, 2))
        feature_seq_reduced = feature_seq_reduced.reshape(batch_size, -1)
        classified_seqwise = self.seq_wise_classifier_net2(feature_seq_reduced)
        assert classified_seqwise.shape == torch.Size((batch_size, self.num_skills))

        feature_seq = feature_seq.detach()\
            .transpose(-2, -3).reshape(batch_size, 100, 32 * 2)
        assert feature_seq.shape == torch.Size((batch_size, 100, 64))

        return classified_seqwise, feature_seq


class CnnStepwiseSteqwiseClassifierTwoDBase(BaseNetwork):

    def __init__(self,
                 cnn_classifier_net: StepwiseSeqwiseCnnEncoder,
                 dropout=0.0,
                 pos_encoder_variant='transformer',
                 ):
        super(CnnStepwiseSteqwiseClassifierTwoDBase, self).__init__()

        self.cnn_encoder = cnn_classifier_net
        pos_enc_ret_dict = create_pos_encoder(
            feature_dim=32*2,
            seq_len=100,
            pos_encoder_variant='transformer',
        )
        self.pos_encoder = pos_enc_ret_dict['pos_encoder']

    def forward(self, seq):
        classified_seqs, feature_seq = self.cnn_encoder(seq)
        pos_encoded_feature_seq = self.pos_encoder(feature_seq)

        classified_steps = self.cnn_encoder.stepwise_classifier_net(pos_encoded_feature_seq)

        return 0



