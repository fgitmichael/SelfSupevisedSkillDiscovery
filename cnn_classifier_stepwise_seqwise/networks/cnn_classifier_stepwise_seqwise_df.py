import torch

from cnn_classifier_stepwise_seqwise.base.cnn_classifier_stepwise_seqwise_df_base import \
    CnnStepwiseSeqwiseClassifierDfBase

from self_supervised.network.flatten_mlp import FlattenMlpDropout


class CnnStepwiseSeqwiseClassifierDiscrete(CnnStepwiseSeqwiseClassifierDfBase):

    @property
    def num_skills(self):
        return self.skill_dim

    def create_stepwise_classifier(self,
                                   feature_dim,
                                   skill_dim,
                                   hidden_sizes,
                                   dropout=0.,
                                   ):
        return FlattenMlpDropout(
            input_size=feature_dim,
            output_size=skill_dim,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

    def check_obs_seq(self, obs):
        seq_dim = 1
        data_dim = -1
        assert len(obs.shape) == 3
        assert obs.size(seq_dim) > obs.size(data_dim)

    def forward(self,
                obs_next,
                train=False,
                ):
        feature_seq, feature_seq_raw = self._process_seq(obs_next)
        feature_seq_pos_enc = self.pos_encoder(feature_seq.detach())

        pred_skill_step = self.stepwise_classifier(feature_seq_pos_enc)
        pred_skill_seq = self.seqwise_classifier(feature_seq_raw)

        classified_steps = pred_skill_step
        classified_seqs = pred_skill_seq

        if train:
            return classified_steps, classified_seqs

        else:
            return classified_steps
