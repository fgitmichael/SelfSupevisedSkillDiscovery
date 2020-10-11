import torch
import torch.nn as nn

from latent_with_splitseqs.networks.seqwise_splitseq_classifier_rnn_whole_seq_recon \
    import SeqwiseSplitseqClassifierRnnWholeSeqRecon
from latent_with_splitseqs.networks.seqwise_splitseq_classifier_rnn_end_recon_only \
    import SeqwiseSplitseqClassifierRnnEndReconOnly

from latent_with_splitseqs.networks.seqwise_splitseq_classifier_latent_seq_end_recon \
    import SeqwiseSplitseqClassifierSlacLatentSeqEndOnlyRecon
from latent_with_splitseqs.networks.seqwise_splitseq_classifier_latent_whole_seq_recon \
    import SeqwiseSplitseqClassifierSlacLatentWholeSeqRecon

from latent_with_splitseqs.latent.rnn_dim_wise import GRUDimwise

from latent_with_splitseqs.networks.seqwise_splitseq_classifier_latent_whole_seq_recon \
    import SeqwiseSplitseqClassifierSlacLatentWholeSeqRecon
from latent_with_splitseqs.networks.seqwise_splitseq_classifier_latent_seq_end_recon \
    import SeqwiseSplitseqClassifierSlacLatentSeqEndOnlyRecon

from latent_with_splitseqs.networks.seqwise_splitseq_classifier_rnn_end_recon_only \
    import SeqwiseSplitseqClassifierRnnEndReconOnly
from latent_with_splitseqs.networks.seqwise_splitseq_classifier_latent_whole_seq_recon \
    import SeqwiseSplitseqClassifierSlacLatentWholeSeqRecon


def get_df(
        obs_dim,
        seq_len,
        skill_dim,
        df_kwargs,
        df_type,
):
    keys = dict(
        feature_extractor='feature_extractor',
        recon='recon',
        rnn_type='rnn_type',
    )

    feature_extractor_types = dict(
        rnn='rnn',
        latent='latent'
    )

    recon_types = dict(
        whole_seq='whole_seq',
        end_only='end_only'
    )

    rnn_type = dict(
        normal='normal',
        dim_wise='dim_wise',
    )

    assert keys['feature_extractor'] in df_kwargs[keys['type']].keys()

    if df_type[keys['feature_extractor']] \
        == feature_extractor_types['rnn']:

        # RNN type
        if df_type[keys['type']][keys['rnn_type']] == rnn_type['normal']:
            rnn = nn.GRU(
                input_size=obs_dim,
                hidden_size=df_kwargs['hidden_size_rnn'],
                batch_first=True,
                bidirectional=False,
            )

        elif df_type[keys['rnn_type']] == rnn_type['dim_wise']:
            rnn = GRUDimwise(
                input_size=obs_dim,
                hidden_size=df_kwargs['hidden_size_rnn'],
                batch_first=True,
                bidirectional=False,
                out_feature_size=df_kwargs['hidden_size_rnn']
            )

        else:
            raise NotImplementedError

        # Classifier using rnn from above
        if df_type[keys['recon']] == recon_types['whole_seq']:
            df = SeqwiseSplitseqClassifierRnnWholeSeqRecon(
                seq_len=seq_len,
                obs_dim=obs_dim,
                skill_dim=skill_dim,
                rnn=rnn,
                **df_kwargs,
            )

        elif df_type[keys['recon']] == recon_types['end_only']:
            df = SeqwiseSplitseqClassifierRnnEndReconOnly(
                seq_len=seq_len,
                obs_dim=obs_dim,
                skill_dim=skill_dim,
                rnn=rnn,
                **df_kwargs,
            )

        else:
            raise NotImplementedError


    elif df_type[keys['type']] == feature_extractor_types['latent']:
        raise NotImplementedError




