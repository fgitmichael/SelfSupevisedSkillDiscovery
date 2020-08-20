from torch import distributions

from seqwise_cont_skillspace.base.rnn_classifier_base import \
    StepwiseSeqwiseClassifierBase

from self_supervised.network.flatten_mlp import FlattenMlpDropout


class StepwiseRnnClassifier(StepwiseSeqwiseClassifierBase):

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
    ) -> FlattenMlpDropout:
        return FlattenMlpDropout(
            input_size=feature_dim,
            output_size=skill_dim,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

    def classify_seqwise(self, data):
        raise NotImplementedError

    def classify_stepwise(self, hidden_seq_stacked):
        """
        Args:
            hidden_seq_stacked          : (N * S, 2 * hidden_size_rnn)
        Return:
            pred_skills                 : (N * S, skill_dim)

        """
        assert hidden_seq_stacked.size(self.data_dim) == \
               self.rnn_params['num_features_hs_posenc']
        return_dict = self.classifier_step(hidden_seq_stacked, train=True)

        return return_dict

    def forward(self,
                seq_batch,
                train=False):
        hidden_seq, _ = self._process_seq(seq_batch)
        hidden_seq = self.pos_encoder(hidden_seq)

        batch_size = hidden_seq.size(self.batch_dim)
        seq_len = hidden_seq.size(self.seq_dim)
        hidden_seq_dim = hidden_seq.size(self.data_dim)

        pred_skill_steps_dict = self.classify_stepwise(
            hidden_seq_stacked=hidden_seq.
                reshape(batch_size * seq_len, hidden_seq_dim)
        )

        if train:
            return dict(
                classified_steps=pred_skill_steps_dict['post'],
                feature_recon_dist=pred_skill_steps_dict['recon'],
                hidden_features_seq=hidden_seq,
            )
        else:
            return distributions.Normal(
                loc=pred_skill_steps_dict['post']['dist'].loc.reshape(
                    seq_batch.size(self.batch_dim),
                    seq_batch.size(self.seq_dim),
                    self.skill_dim),
                scale=pred_skill_steps_dict['post']['dist'].scale.reshape(
                    seq_batch.size(self.batch_dim),
                    seq_batch.size(self.seq_dim),
                    self.skill_dim)
            )
