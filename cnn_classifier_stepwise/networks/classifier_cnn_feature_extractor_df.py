from cnn_classifier_stepwise.base.cnn_classifier_stepwise_base import \
    CnnStepwiseClassifierBaseDf

from self_supervised.network.flatten_mlp import FlattenMlpDropout

class CnnStepwiseClassifierDiscreteDf(CnnStepwiseClassifierBaseDf):

    def create_stepwise_classifier(self,
                                   feature_dim,
                                   skill_dim,
                                   hidden_sizes,
                                   dropout=0.,
                                   ) -> FlattenMlpDropout:
        return FlattenMlpDropout(
            input_size=feature_dim,
            output_size=skill_dim,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

    @property
    def num_skills(self):
        return self.skill_dim

    def forward(self,
                obs_next,
                train=False,
                ):
        """
        Args:
            obs_next                    : (N ,S, data_dim)
            train                       : bool
        Return:
            train is True
                pred_skills_step        : (N, S, num_skills)
                feature_seq             : (N, S, feature_dim)
            train is False
                pred_skills_step        : (N, S, num_skills)
        """
        feature_seq = self._process_seq(obs_next)
        feature_seq_pos_enc = self.pos_encoder(feature_seq)

        pred_skill_scores = self.stepwise_classifier(feature_seq_pos_enc)

        if train:
            return dict(
                classified_steps=pred_skill_scores,
                hidden_features_seq=feature_seq,
            )

        else:
            return pred_skill_scores
