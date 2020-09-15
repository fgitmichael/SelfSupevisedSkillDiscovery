from diayn_seq_code_revised.networks.bi_rnn_stepwise_seqwise_singledims import \
    BiRnnStepwiseSeqWiseClassifierSingleDims

from diayn_seq_code_revised.networks.my_gaussian \
    import MyGaussian


class BiRnnStepwiseSeqWiseClassifierSingleDimsContOutput(
    BiRnnStepwiseSeqWiseClassifierSingleDims):
    """
    This class is used to make the classifier compatible
    with ContSkillTrainerSeqwiseStepwise and it's subclasses
    """
    def __init__(self,
                 *args,
                 dropout,
                 skill_dim,
                 feature_decode_hidden_size=(256, 256),
                 layer_norm=False,
                 **kwargs,
                 ):
        super().__init__(*args,
                         dropout=dropout,
                         skill_dim=skill_dim,
                         **kwargs)

        self.feature_decoder = MyGaussian(
            input_dim=skill_dim,
            output_dim=self.pos_enc_feature_size,
            hidden_units=feature_decode_hidden_size,
            layer_norm=layer_norm,
        )

    def create_step_classifier(self,
                               input_size,
                               output_size,
                               hidden_sizes,
                               dropout,
                               layer_norm=False,
                               ):
        return MyGaussian(
            input_dim=input_size,
            output_dim=output_size,
            hidden_units=hidden_sizes,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def forward(self, seq_batch, train=False):
        """
        Args:
            seq_batch           : (N, S, data_dim)
        Return:
            classified_steps    : (N, S, skill_dim)
            feature_recon_dist  : (N, S, pos_enc_feature_size)
            classified_seqs     : (N, skill_dim)
            features            : (N, S, pos_enc_feature_size)
        """
        assert len(seq_batch.shape) == 3

        hidden_seqs, h_n_s = self._process_seq(seq_batch)
        hidden_seqs = hidden_seqs.detach()
        hidden_seqs_feature_matched = self.hidden_features_dim_matcher(hidden_seqs)
        hidden_seqs_feature_matched_pos_enc = self.pos_encoder(
            hidden_seqs_feature_matched)

        classified_steps = self.classifier(hidden_seqs_feature_matched_pos_enc)
        classified_seqs = self.classifier_seq(h_n_s)

        # Decode skill to feature (:= classified steps)
        feature_recon_dist = self.feature_decoder(classified_steps.rsample())

        if train:
            return dict(
                classified_steps=classified_steps,
                feature_recon_dist=feature_recon_dist,
                classified_seqs=classified_seqs,
                hidden_features_seq=hidden_seqs_feature_matched_pos_enc,
            )

        else:
            return classified_steps
