from diayn_seq_code_revised.networks.bi_rnn_stepwise_seqwise_noid import \
    BiRnnStepwiseSeqwiseNoidClassifier
from diayn_seq_code_revised.networks.my_gaussian import MyGaussian as Gaussian

from diayn_rnn_seq_rnn_stepwise_classifier.networks.bi_rnn_stepwise import \
    BiRnnStepwiseClassifier


class RnnVaeClassifierContSkills(BiRnnStepwiseSeqwiseNoidClassifier):

    def __init__(self,
                 *args,
                 feature_decode_hidden_size=(256, 256),
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.feature_decoder = Gaussian(
            input_dim=self.classifier.output_size,
            output_dim=2 * self.rnn.hidden_size,
            hidden_units=feature_decode_hidden_size
        )

    def forward(self,
                seq_batch,
                train=False):
        """
        Args:
            seq_batch           : (N, S, data_dim)
        Return:
            classified_steps    : (N, S, num_skills)
            feature_recon_dist  : (N, S, hidden_size_rnn)
            classified_seqs     : (N, num_skills)
            hidden_features_seq : (N, S, hidden_size_rnn)

        Change to base method: hidden_features_seq is returned
        """
        assert len(seq_batch.shape) == 3

        # Call higher base method to get hidden features seq
        classified_steps, \
        hidden_features_seq, \
        h_n = BiRnnStepwiseClassifier.forward(
            self,
            seq_batch=seq_batch,
            return_rnn_outputs=True
        )

        # Decode skill to feature (:= classified steps)
        feature_recon_dist = self.feature_decoder(classified_steps.rsample())

        classified_seqs = self._classify_seq_seqwise(h_n)

        if train:
            return dict(
                classified_steps=classified_steps,
                feature_recon_dist=feature_recon_dist,
                classified_seqs=classified_seqs,
                hidden_features_seq=hidden_features_seq
            )

        else:
            return classified_steps
