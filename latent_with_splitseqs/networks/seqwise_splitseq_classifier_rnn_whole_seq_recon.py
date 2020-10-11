from latent_with_splitseqs.networks.seqwise_splitseq_classifier_rnn_end_recon_only import \
    SeqwiseSplitseqClassifierRnnEndReconOnly

from self_supervised.utils.my_pytorch_util import tensor_equality


class SeqwiseSplitseqClassifierRnnWholeSeqRecon(
    SeqwiseSplitseqClassifierRnnEndReconOnly):

    def train_forwardpass(self,
                          obs_seq,
                          skill,):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size, seq_len, obs_dim = obs_seq.shape

        hidden_seq, _ = self.rnn(obs_seq)
        hidden_seq_reshaped = hidden_seq.reshape(
            batch_size * seq_len,
            hidden_seq.size(data_dim)
        )
        skill_recon_dist = self.classifier(
            hidden_seq_reshaped
        )
        # TODO: delete assertion
        assert tensor_equality(hidden_seq_reshaped[0], hidden_seq[0, 0, :])

        return dict(
            skill_recon_dist=skill_recon_dist
        )
