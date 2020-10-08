import torch

from latent_with_splitseqs.networks.seqwise_splitseq_classifier_whole_seq_recon \
    import SeqwiseSplitseqClassifierSlacLatentWholeSeqRecon


class SeqwiseSplitseqClassifierSlacLatentSeqEndOnlyRecon(
    SeqwiseSplitseqClassifierSlacLatentWholeSeqRecon):

    def train_forwardpass(self,
                          obs_seq,
                          skill,):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = obs_seq.size(batch_dim)
        seq_len = obs_seq.size(seq_dim)

        pri_post_dict = self.latent_net(
            skill=skill,
            obs_seq=obs_seq,
        )

        pri = pri_post_dict['pri']
        post = pri_post_dict['post']
        latent_seq = torch.cat(
            [post['latent1_samples'],
             post['latent2_samples']],
            dim=data_dim
        )

        skill_recon_dist = self.classifier(latent_seq[:, -1, :])
        assert skill_recon_dist.batch_shape == skill.shape

        return dict(
            latent_pri=pri,
            latent_post=post,
            recon=skill_recon_dist,
        )


