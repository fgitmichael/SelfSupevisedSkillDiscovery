from seqwise_cont_skillspace.base.rnn_classifier_base\
    import RnnStepwiseSeqwiseClassifierBase

from self_supervised.network.flatten_mlp import FlattenMlpDropout


class StepwiseOnlyRnnClassifierDiscrete(RnnStepwiseSeqwiseClassifierBase):

    def __init__(self,
                 *args,
                 skill_dim,
                 **kwargs):
        super(StepwiseOnlyRnnClassifierDiscrete, self).__init__(
            *args,
            skill_dim=skill_dim,
            **kwargs
        )
        self.num_skills = skill_dim

    def create_seqwise_classifier(
            self,
            feature_dim,
            skill_dim,
            hidden_sizes,
            dropout=0.,
    ):
        return None

    def create_stepwise_classifier(
            self,
            feature_dim,
            skill_dim,
            hidden_sizes,
            dropout=0.
    ):
        return FlattenMlpDropout(
            input_size=feature_dim,
            output_size=skill_dim,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

    def classify_seqwise(self, data):
        raise NotImplementedError("It's a stepwise only classifier!")

    def classify_stepwise(self, hidden_seq):
        """
        Args:
            hidden_seq              : (N, S, 2 * hidden_size_rnn)
        Return:
            ret                     : (N, S, num_skills) classification scores
        """
        return self.classifier_step(hidden_seq)

    def forward(self,
                obs_next,
                train=False):
        """
        Args:
            obs_next                : (N, S, data_dim)
        Return:
            train==True:
                pred_skills_step    : (N, S, num_skills)
                hidden_features_seq : (N, S, 2*hidden_size_rnn)
            train==False:
                pred_skills_step    : (N, S, num_skills)
        """
        hidden_seq, _ = self._process_seq(obs_next)
        hidden_seq = self.pos_encoder(hidden_seq)

        pred_skill_scores = self.classify_stepwise(
            hidden_seq=hidden_seq
        )

        if train:
            return dict(
                classified_steps=pred_skill_scores,
                hidden_features_seq=hidden_seq
            )
        else:
            return pred_skill_scores
