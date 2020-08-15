from self_supervised.network.flatten_mlp import FlattenMlp

from seqwise_cont_skillspace.base.rnn_classifier_base import \
    StepwiseSeqwiseClassifierBase
from diayn_original_cont.networks.vae_regressor import VaeRegressor


class StepwiseSeqwiseClassifierVae(StepwiseSeqwiseClassifierBase):

    def __init__(self,
                 *args,
                 skill_dim,
                 **kwargs):
        super(StepwiseSeqwiseClassifierVae, self).__init__(*args, **kwargs)
        self.skill_dim = skill_dim

    def create_seqwise_classifier(
            self,
            feature_dim,
            skill_dim,
            hidden_sizes
    ) -> FlattenMlp:
        return FlattenMlp(
            input_size=feature_dim,
            output_size=skill_dim,
            hidden_sizes=hidden_sizes,
        )

    def create_stepwise_classifier(
            self,
            feature_dim,
            skill_dim,
            hidden_sizes
    ) -> VaeRegressor:
        return VaeRegressor(
            input_size=feature_dim,
            latent_dim=self.skill_dim,
            output_size=feature_dim,
            hidden_sizes_enc=hidden_sizes,
            hidden_sizes_dec=hidden_sizes
        )

    def classify_seqwise(self, h_n):
        """
        Args:
            h_n                 : (N, num_features)
        Return:
            pred_skill          : (N, skill_dim)
        """
        assert self.classifier_step.input_size == self.rnn_params['num_features']
        assert h_n.size(self.data_dim) == self.rnn_params['num_features']

        return self.classifier_step(h_n)

    def classify_stepwise(self, hidden_seq_stacked):
        """
        Args:
            hidden_seq_stacked          : (N * S, 2 * hidden_size_rnn)
        Return:
            pred_skills         : (N * S, skill_dim)

        """
        assert hidden_seq_stacked.size(self.data_dim) == self.rnn_params['num_channels']
        return_dict = self.classifier_step(hidden_seq_stacked)

        return return_dict

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
            h_n=h_n
        )

        hidden_seq = hidden_seq.detach()

        batch_size = hidden_seq.size(self.batch_dim)
        seq_len = hidden_seq.size(self.seq_dim)
        hidden_seq_dim = hidden_seq.size(self.data_dim)
        pred_skill_steps_dict = self.classify_stepwise(
            hidden_seq_stacked=hidden_seq.
                view(batch_size * seq_len, hidden_seq_dim)
        )

        if train:
            return dict(
                classified_steps=pred_skill_steps_dict['post'],
                feature_recon_dist=pred_skill_steps_dict['recon'],
                classified_seqs=pred_skill_seq,
                hidden_features_seq=hidden_seq,
            )
        else:
            return pred_skill_steps_dict['post']
