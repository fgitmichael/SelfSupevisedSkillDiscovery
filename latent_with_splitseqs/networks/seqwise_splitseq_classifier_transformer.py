import torch
import torch.nn as nn
import torch.distributions as torch_dist

from diayn_seq_code_revised.networks.my_gaussian import MyGaussian as Gaussian

from latent_with_splitseqs.base.classifier_base import SplitSeqClassifierBase
from latent_with_splitseqs.base.my_object_base import MyObjectBase

from latent_with_splitseqs.latent.df_transformer import DfTransformer


class SeqwiseSplitseqClassifierTransformer(SplitSeqClassifierBase, MyObjectBase):

    def __init__(
            self,
            *args,
            seq_len,
            obs_dim,
            skill_dim,
            df_transformer: DfTransformer,
            hidden_units_classifier=(256, 256),
            std_classifier,
            dropout,
            leaky_slope_classifier,
            **kwargs
    ):
        super().__init__(
            *args,
            obs_dim=obs_dim,
            seq_len=seq_len,
            **kwargs
        )

        self.transformer_encoder = df_transformer

        self.classifier = Gaussian(
            input_dim=df_transformer.output_size * seq_len,
            output_dim=skill_dim,
            hidden_units=hidden_units_classifier,
            leaky_slope=leaky_slope_classifier,
            dropout=dropout,
            std=std_classifier,
        )

        self.obs_dim = obs_dim
        self.skill_dim = skill_dim

    @property
    def _objs_to_save(self):
        objs_to_save = super()._objs_to_save
        return dict(
            **objs_to_save,
            transformer_encoder=self.transformer_encoder,
            classifier=self.classifier,
        )

    def _reshape_transformer_encoding(self, transformer_encoding: torch.Tensor):
        """
        Args:
            transformer_encoding                : (batch_size, seq_len, data_dim)
        Returns:
            out                                 : (batch_size, seq_len * data_dim)
        """
        batch_dim, seq_dim, data_dim = 0, 1, 2
        batch_size = transformer_encoding.shape[batch_dim]
        seq_len = transformer_encoding.shape[seq_dim]
        obs_dim = transformer_encoding.shape[data_dim]

        transformer_encoding = transformer_encoding.permute(seq_dim, data_dim, batch_dim)
        batch_dim, seq_dim, data_dim = 2, 0, 1
        transformer_encoded = torch.reshape(
            transformer_encoding,
            (seq_len * obs_dim,
             batch_size)
        )
        transformer_encoding = transformer_encoded.permute(-1, 0)
        assert transformer_encoding.shape == torch.Size((batch_size, seq_len * obs_dim))

        return transformer_encoding

    @torch.no_grad()
    def eval_forwardpass(
            self,
            obs_seq: torch.Tensor,
            skill: torch.Tensor,
            **kwargs,
    ):
        transformer_encoded = self.transformer_encoder(obs_seq)
        transformer_encoded_reshaped = self._reshape_transformer_encoding(transformer_encoded)
        skill_recon_dist = self.classifier(transformer_encoded_reshaped)

        return dict(
            skill_recon_dist=skill_recon_dist,
            feature_seq=transformer_encoded,
        )

    def train_forwardpass(
            self,
            obs_seq,
            skill,
    ) -> dict:
        """
        Args:
            obs_seq                 : (N, S, obs_dim)
            skill                   : (N, skill_dim)
        Returns:
            out                     : dict
                skill_recon_dist    : (N, skill_dim) distribution
        """
        transformer_encoded = self.transformer_encoder(obs_seq)
        transformer_encoded = self._reshape_transformer_encoding(transformer_encoded)
        skill_recon_dist = self.classifier(transformer_encoded)

        batch_dim, seq_dim, data_dim = 0, 1, -1
        assert isinstance(skill_recon_dist, torch_dist.Distribution)
        assert skill_recon_dist.batch_shape \
            == torch.Size((obs_seq.shape[batch_dim], skill.shape[data_dim]))

        return dict(
            skill_recon_dist=skill_recon_dist,
        )
