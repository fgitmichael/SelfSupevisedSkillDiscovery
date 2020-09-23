from torch import distributions
import warnings

from seqwise_cont_skillspace.base.rnn_classifier_base import \
    RnnStepwiseSeqwiseClassifierBase

from self_supervised.network.flatten_mlp import FlattenMlp, FlattenMlpDropout

from diayn_original_cont.networks.vae_regressor import VaeRegressor

from diayn_seq_code_revised.networks.bi_rnn_stepwise_seqwise_singledims \
    import BiRnnStepwiseSeqWiseClassifierSingleDims


class RnnStepwiseSeqwiseClassifierHduvaeSingleDims(
    BiRnnStepwiseSeqWiseClassifierSingleDims):

    def __init__(self,
                 *args,
                 skill_dim,
                 dropout,
                 hidden_sizes_classifier_step,
                 hidden_sizes_feature_to_latent_encoder,
                 hidden_sizes_latent_to_skill_decoder,
                 layer_norm=False,
                 **kwargs):
        super().__init__(
            *args,
            skill_dim=skill_dim,
            dropout=dropout,
            hidden_sizes_classifier_step=hidden_sizes_classifier_step,
            layer_norm=layer_norm,
            **kwargs)

        # Overwrite Step Classifier
        if hidden_sizes_classifier_step is not None:
            warnings.warn("hidden_sizes_classifier_step - argument is not used."
                          "Input None instead!")
        hidden_sizes_step_classifier = dict(
            enc=hidden_sizes_feature_to_latent_encoder,
            dec=hidden_sizes_latent_to_skill_decoder,
        )
        self.classifier = self.create_step_classifier(
            input_size=self.pos_enc_feature_size,
            output_size=skill_dim,
            hidden_sizes=hidden_sizes_step_classifier,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def create_step_classifier(self,
                               input_size,
                               output_size,
                               hidden_sizes,
                               dropout,
                               layer_norm=False,
                               ) -> VaeRegressor:
        if hidden_sizes is not None:
            return VaeRegressor(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes_enc=hidden_sizes['enc'],
                hidden_sizes_dec=hidden_sizes['dec'],
                std=None,
                latent_dim=2,
            )
        else:
            return None

    def forward(
            self,
            seq_batch,
            train=False
    ):
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
        hidden_seqs, h_n_s = self._process_seq(seq_batch)
        hidden_seqs = hidden_seqs.detach()
        hidden_seqs_feature_matched = self.hidden_features_dim_matcher(hidden_seqs)
        hidden_seqs_feature_matched_pos_enc = self.pos_encoder(
            hidden_seqs_feature_matched
        )

        hidden_seqs_feature_matched_pos_enc_stacked = \
            hidden_seqs_feature_matched_pos_enc.reshape(
                -1, hidden_seqs_feature_matched_pos_enc.size(-1))
        pred_skill_steps_dict = self.classifier(
            hidden_seqs_feature_matched_pos_enc_stacked,
            train=True,
        )
        pred_skill_seq = self.classifier_seq(h_n_s)

        if train:
            return dict(
                skill_recon=pred_skill_steps_dict['recon'],
                latent_post=pred_skill_steps_dict['post'],
                classified_seqs=pred_skill_seq,
                hidden_features_seq=hidden_seqs,
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

