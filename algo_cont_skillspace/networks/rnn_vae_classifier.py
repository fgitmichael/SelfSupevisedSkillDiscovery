from diayn_seq_code_revised.networks.bi_rnn_stepwise_seqwise_noid import \
    BiRnnStepwiseSeqwiseNoidClassifier

from diayn_rnn_seq_rnn_stepwise_classifier.networks.bi_rnn_stepwise import \
    BiRnnStepwiseClassifier


class RnnVaeClassifierContSkills(BiRnnStepwiseSeqwiseNoidClassifier):

    def forward(self,
                seq_batch,
                train=False):
        """
        Args:
            seq_batch           : (N, S, data_dim)
        Return:
            classified_steps    : (N, S, num_skills)
            classified_seqs     : (N, num_skills)
            hidden_features_seq : (N, S, hidden_size_rnn)

        Change to base method: hidden_features_seq is returned
        """
        assert len(seq_batch.shape) == 3

        classified_steps, \
        hidden_features_seq, \
        h_n = BiRnnStepwiseClassifier.forward(
            self,
            seq_batch=seq_batch,
            return_rnn_outputs=True
        )

        classified_seqs = self._classify_seq_seqwise(h_n)

        if train:
            return dict(
                classified_steps=classified_steps,
                classified_seqs=classified_seqs,
                hidden_features_seq=hidden_features_seq
            )

        else:
            return classified_steps











