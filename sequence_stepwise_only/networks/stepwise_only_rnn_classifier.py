from seqwise_cont_skillspace.networks.rnn_seqwise_stepwise_classifier_revised \
    import StepwiseSeqwiseClassifierBase

from diayn_original_cont.networks.vae_regressor import VaeRegressor


class StepwiseOnlyRnnClassifierCont(StepwiseSeqwiseClassifierBase):

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
    ) -> VaeRegressor:
        return VaeRegressor(
            input_size=feature_dim,
            latent_dim=skill_dim,
            output_size=feature_dim,
            hidden_sizes_enc=hidden_sizes,
            hidden_sizes_dec=hidden_sizes,
            dropout=dropout,
        )

    def classify_seqwise(self, data):
        raise NotImplementedError("It's a stepwise only classifier!")

    def classify_stepwise(self, hidden_seq):
        """
        Args:
            hidden_seq              : (N, S, 2 * hidden_size_rnn)
        Return:
            return_dict of VAE-regressor
        """
        return_dict = self.classifier_step(hidden_seq, train=True)
        return return_dict

    def forward(self,
                obs_next,
                train=False):

        """
        Args:
            obs_next                    : (N, S, data_dim)
        Return:
            train==True:
                pred_skills_step        : (N, S, skill_dim)
            train==False:
                pred_skills_step        : (N, skill_dim) distribution
        """
        hidden_seq, h_n = self._process_seq(obs_next)
        hidden_seq = self.pos_encoder(hidden_seq)

        pred_skill_dict_step = self.classify_stepwise(
            hidden_seq=hidden_seq
        )

        if train:
            return dict(
                classified_steps=pred_skill_dict_step['post'],
                feature_recon_dist=pred_skill_dict_step['recon'],
                hidden_features_seq=hidden_seq
            )

        else:
            return pred_skill_dict_step['post']['dist']
