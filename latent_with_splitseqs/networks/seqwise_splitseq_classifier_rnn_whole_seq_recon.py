from latent_with_splitseqs.networks.seqwise_splitseq_classifier_rnn_end_recon_only import \
    SeqwiseSplitseqClassifierRnnEndReconOnly


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
        skill_recon_dist = self.classifier(
            hidden_seq.reshape(
                batch_size * seq_len,
                obs_dim,
            )
        )
        # TODO: delete assertion
        assert hidden_seq.reshape(
            batch_size * seq_len,
            obs_dim,
        )[0] == hidden_seq[0, 0, :]

        return dict(
            skill_recon_dist=skill_recon_dist
        )
