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

df_type_keys = dict(
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

rnn_types = dict(
    normal='normal',
    dim_wise='dim_wise',
)


def get_df(
        obs_dim,
        seq_len,
        skill_dim,
        df_kwargs,
        df_type,
):
    """
    Args:
        obs_dim                     : observation dimensionality
        seq_len                     : sequence length
        skill_dim                   : skill dimensionality
        df_kwargs
            obs_dims_used_df        : list or tuple
            dropout                 : float
            hidden_size_rnn         : int
            leaky_slop_classifier   : float
            hidden_units_classifier : list or tuple
        df_type
            feature_extractor       : which method to extract sequential features
            recon                   : full seq reconstruction or end reconstruction

    """
    global df_type_keys
    global feature_extractor_types
    global recon_types
    global rnn_types

    if df_type[df_type_keys['feature_extractor']] \
        == feature_extractor_types['rnn']:

        # RNN type
        if df_type[df_type_keys['rnn_type']] == rnn_types['normal']:
            rnn = nn.GRU(
                input_size=obs_dim,
                hidden_size=df_kwargs['hidden_size_rnn'],
                batch_first=True,
                bidirectional=False,
            )

        elif df_type[df_type_keys['rnn_type']] == rnn_types['dim_wise']:
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
        if df_type[df_type_keys['recon']] == recon_types['whole_seq']:
            df = SeqwiseSplitseqClassifierRnnWholeSeqRecon(
                seq_len=seq_len,
                obs_dim=obs_dim,
                skill_dim=skill_dim,
                rnn=rnn,
                **df_kwargs,
            )

        elif df_type[df_type_keys['recon']] == recon_types['end_only']:
            df = SeqwiseSplitseqClassifierRnnEndReconOnly(
                seq_len=seq_len,
                obs_dim=obs_dim,
                skill_dim=skill_dim,
                rnn=rnn,
                **df_kwargs,
            )

        else:
            raise NotImplementedError


    elif df_type[df_type_keys['type']] == feature_extractor_types['latent']:
        raise NotImplementedError

    else:
        raise NotImplementedError

    return df




