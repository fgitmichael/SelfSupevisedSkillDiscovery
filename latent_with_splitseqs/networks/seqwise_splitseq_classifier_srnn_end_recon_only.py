import torch

from latent_with_splitseqs.networks.seqwise_splitseq_classifier_srnn_whole_seq_recon \
    import SplitSeqClassifierSRNNWholeSeqRecon


class SplitSeqClassifierSRNNEndReconOnly(SplitSeqClassifierSRNNWholeSeqRecon):

    def train_forwardpass(
            self,
            obs_seq,
            skill,
    ):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = obs_seq.size(batch_dim)
        seq_len = obs_seq.size(seq_dim)

        pri_post_dict = self.latent_net(
            skill=skill,
            obs_seq=obs_seq,
        )

        pri = pri_post_dict['pri_dict']
        post = pri_post_dict['post_dict']
        latent_seq = post['latent_samples'][:, -1, :]

        skill_recon_dist = self.classifier(latent_seq)
        assert skill_recon_dist.batch_shape[:data_dim] \
               == torch.Size((batch_size, ))
        assert skill_recon_dist.batch_shape[data_dim] == skill.shape[data_dim]

        return dict(
            latent_pri=pri,
            latent_post=post,
            recon=skill_recon_dist,
        )
