from self_supervised.network.flatten_mlp import FlattenMlp

from seqwise_cont_skillspace.base.rnn_classifier_base import \
    StepwiseSeqwiseClassifierBase


class StepwiseSeqwiseClassifierStandard(StepwiseSeqwiseClassifierBase):

    def create_seqwise_classifier(
            self,
            input_size,
            output_size,
            hidden_sizes
    ) -> FlattenMlp:
        return FlattenMlp(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
        )

    def create_stepwise_classifier(
            self,
            input_size,
            output_size,
            hidden_sizes,
    ) -> FlattenMlp:
        return FlattenMlp(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
        )

    def classify_seqwise(self, h_n):
        """
        Args:
            h_n                 : (N, num_features)
        Return:
            pred_skill          : (N, skill_dim)
        """
        assert self.step_classifier.input_size == self.rnn_params['num_features']
        assert h_n.size(self.data_dim) == self.rnn_params['num_features']

        return self.step_classifier(h_n)

    def classify_stepwise(self, hidden_seq):
        """
        Args:
            hidden_seq          : (N * S, 2 * hidden_size_rnn)
        Return:
            pred_skills         : (N * S, skill_dim)

        """
        assert hidden_seq.size(self.data_dim) == self.rnn_params['num_channels']
        pred_skills = self.step_classifier(hidden_seq)

        return pred_skills

    def forward(self,
                seq_batch,
                train=False):
        """
        Args:
            seq_batch                   : (N, S, data_dim)
        Return:
            train==True:
                pred_skills_step        : (N, S, skill_dim)
                pred_skills_seq         : (N, skill_dim)
            train==False:
                pred_skills_step        : (N, skill_dim)
        """
        hidden_seq, h_n = self._process_seq(seq_batch)

        pred_skill_seq = self.classify_seqwise(
            h_n=h_n)

        pred_skill_steps = self.classify_stepwise(
            hidden_seq=hidden_seq.detach()
        )

        if train:
            return pred_skill_steps, pred_skill_seq
        else:
            return pred_skill_steps
