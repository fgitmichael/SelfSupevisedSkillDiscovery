import torch
import torch.nn as nn

from code_slac.network.base import BaseNetwork
#from code_slac.network.latent import Gaussian

from diayn_seq_code_revised.networks.my_gaussian import MyGaussian as Gaussian

from latent_with_splitseqs.base.classifier_base import SplitSeqClassifierBase
from latent_with_splitseqs.base.my_object_base import MyObjectBase


class SeqwiseSplitseqClassifierRnnEndReconOnly(SplitSeqClassifierBase, MyObjectBase):

    def __init__(self,
                 *args,
                 seq_len,
                 obs_dim,
                 skill_dim,
                 rnn: nn.RNNBase,
                 hidden_units_classifier=(256, 256),
                 leaky_slope_classifier=0.2,
                 std_classifier=None,
                 dropout=0.3,
                 **kwargs
                 ):
        super(SeqwiseSplitseqClassifierRnnEndReconOnly, self).__init__(
            *args,
            obs_dim=obs_dim,
            seq_len=seq_len,
            **kwargs
        )

        #self.rnn = nn.GRU(
        #    input_size=obs_dim,
        #    hidden_size=hidden_size_rnn,
        #    batch_first=True,
        #    bidirectional=False,
        #)
        self.rnn = rnn

        self.classifier = Gaussian(
            input_dim=self.rnn.hidden_size,
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
            rnn=self.rnn,
            classifier=self.classifier,
        )

    def process_save_dict(self, save_obj):
        """
        Subclass to be able to apply flatten parameter method,
        otherwise when loading a cuda-rnn object warning occur
        """
        rnn = save_obj.pop('rnn')
        assert isinstance(rnn, nn.RNNBase)
        rnn.flatten_parameters()

        self._set_obj_attr('rnn', rnn)
        super().process_save_dict(save_obj)

    @property
    def seq_len(self):
        return self._seq_len

    @torch.no_grad()
    def eval_forwardpass(self,
                         obs_seq,
                         skill=None):
        hidden_seq, _ = self.rnn(obs_seq)
        skill_recon_dist = self.classifier(hidden_seq[:, -1, :])

        return dict(
            skill_recon_dist=skill_recon_dist,
            feature_seq=hidden_seq,
        )

    def train_forwardpass(self,
                          obs_seq,
                          skill,):
        hidden_seq, _ = self.rnn(obs_seq)
        skill_recon_dist = self.classifier(hidden_seq[:, -1, :])

        return dict(
            skill_recon_dist=skill_recon_dist,
        )
