import torch
from torch import distributions

from seqwise_cont_skillspace.base.rnn_classifier_base import StepwiseSeqwiseClassifierBase

from self_supervised.network.flatten_mlp import FlattenMlp, FlattenMlpDropout

from diayn_original_cont.networks.vae_regressor import VaeRegressor


class SeqwiseStepwiseClassifierContSsvaestyle(StepwiseSeqwiseClassifierBase):

    def create_seqwise_classifier(
            self,
            feature_dim,
            skill_dim,
            hidden_sizes,
            dropout=0.,
    ) -> FlattenMlp:
        return FlattenMlpDropout(
            input_size=feature_dim,
            output_size=skill_dim,
            hidden_sizes=hidden_sizes,
            dropout=dropout
        )

    def create_stepwise_classifier(
            self,
            feature_dim,
            skill_dim,
            hidden_sizes,
            dropout=0.,
    ) -> VaeRegressor:
        return VaeRegressor(
            input_size=feature_dim,
            latent_dim=skill_dim,
            output_size=feature_dim,
            hidden_sizes_enc=hidden_sizes,
            hidden_sizes_dec=hidden_sizes,
            dropout=dropout,
        )

    def classify_seqwise(self, h_n):
        """
        Args:
            h_n             : (N, num_features)
        Return:
            pred_skill      : (N, skill_dim)
        """
        assert self.classifier_seq.input_size == self.rnn_params['num_features_h_n']
        assert h_n.size(self.data_dim) == self.rnn_params['num_features_h_n']

        return self.classifier_seq(h_n)

    def classify_stepwise(self, data_dict):
        """
        Args:
            data_dict
                hidden_seq          : (N, S, 2 * hidden_size_rnn)
                                      detached(!) hidden features
                                      sequence of the rnn
                pred_skills_seq     : (N, skill_dim) predicted skills of seq classifier
        Return:
            pred_skills             : (N, S, skill_dim)
        """
        hidden_seq = data_dict['hidden_seq']

        seq_len = hidden_seq.size(self.seq_dim)
        pred_skills_seq = data_dict['pred_skills_seq']
        pred_skills_seq = torch.stack([pred_skills_seq] * seq_len, dim=self.seq_dim)

        input_classifier = torch.cat([hidden_seq, pred_skills_seq], dim=self.data_dim)
        #assert input_classifier.size(self.data_dim) == self.classifier_step.input_size

        assert hidden_seq.size(self.data_dim) == \
            self.rnn_params['num_features_hs_posenc']
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
                pred_skills_seq         : (N, skill_dim)
            train==False:
                pred_skills_step        : (N, skill_dim)
        """
        hidden_seq, h_n = self._process_seq(obs_next)

        pred_skill_seq = self.classify_seqwise(
            h_n=h_n
        )

        hidden_seq = hidden_seq.detach()
        hidden_seq = self.pos_encoder(hidden_seq)

        pred_skill_dict_step = self.classify_stepwise(
            data_dict=dict(
                hidden_seq=hidden_seq,
                pred_skills_seq=pred_skill_seq.detach(),
            )
        )

        if train:
            return dict(
                classified_steps=pred_skill_dict_step['post'],
                feature_recon_dist=pred_skill_dict_step['recon'],
                classified_seqs=pred_skill_seq,
                hidden_features_seq=hidden_seq,
            )

        else:
            return pred_skill_dict_step['post']['dist']


class GuidedNoSsvaestyle(SeqwiseStepwiseClassifierContSsvaestyle):

    def create_stepwise_classifier(
            self,
            feature_dim,
            skill_dim,
            hidden_sizes,
            dropout=0.,
    ) -> VaeRegressor:
        return VaeRegressor(
            input_size=feature_dim,
            latent_dim=skill_dim,
            output_size=feature_dim,
            hidden_sizes_enc=hidden_sizes,
            hidden_sizes_dec=hidden_sizes,
            dropout=0.5,
        )

    def classify_stepwise(self, data_dict):
        """
        Args:
            data_dict
                hidden_seq          : (N, S, 2 * hidden_size_rnn)
                                      detached(!) hidden features
                                      sequence of the rnn
                pred_skills_seq     : (N, skill_dim) predicted skills of seq classifier
        Return:
            pred_skills             : (N, S, skill_dim)
        """
        hidden_seq = data_dict['hidden_seq']

        seq_len = hidden_seq.size(self.seq_dim)

        assert hidden_seq.size(self.data_dim) == \
               self.rnn_params['num_features_hs_posenc']
        return_dict = self.classifier_step(hidden_seq, train=True)

        return return_dict
