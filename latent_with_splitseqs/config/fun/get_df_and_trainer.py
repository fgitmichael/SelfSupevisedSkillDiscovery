import torch
import torch.nn as nn

from latent_with_splitseqs.networks.seqwise_splitseq_classifier_rnn_whole_seq_recon \
    import SeqwiseSplitseqClassifierRnnWholeSeqRecon
from latent_with_splitseqs.networks.seqwise_splitseq_classifier_rnn_end_recon_only \
    import SeqwiseSplitseqClassifierRnnEndReconOnly

from latent_with_splitseqs.latent.rnn_dim_wise import GRUDimwise, GRUDimwisePosenc
from latent_with_splitseqs.latent.rnn_pos_encoded import GRUPosenc
from latent_with_splitseqs.latent.slac_latent_net_conditioned_on_single_skill \
    import SlacLatentNetConditionedOnSingleSkill
from latent_with_splitseqs.latent.slac_latent_conditioned_on_skill_seq \
    import SlacLatentNetConditionedOnSkillSeq
from latent_with_splitseqs.latent.\
    slac_latent_conditioned_on_skill_seq_smoothing_posterior \
    import SlacLatentNetConditionedOnSkillSeqSmoothingPosterior
from latent_with_splitseqs.latent.one_stochlayered_latent_conditioned_on_skill_seq \
    import OneLayeredStochasticLatent

from latent_with_splitseqs.networks.seqwise_splitseq_classifier_latent_seq_end_recon \
    import SeqwiseSplitseqClassifierSlacLatentSeqEndOnlyRecon
from latent_with_splitseqs.networks.seqwise_splitseq_classifier_latent_whole_seq_recon \
    import SeqwiseSplitseqClassifierSlacLatentWholeSeqRecon
from latent_with_splitseqs.networks.\
    seqwise_splitseq_classifier_latent_singlelayer_whole_seq_recon \
    import SeqwiseSplitseqClassifierSingleLayeredWholeSeqRecon

from latent_with_splitseqs.trainer.latent_with_splitseq_full_seq_recon_loss_trainer \
    import URLTrainerLatentWithSplitseqsFullSeqReconLoss
from latent_with_splitseqs.trainer.latent_with_splitseqs_trainer \
    import URLTrainerLatentWithSplitseqs

from latent_with_splitseqs.trainer.rnn_with_splitseqs_trainer_whole_seq_recon \
    import URLTrainerRnnWithSplitseqsWholeSeqRecon
from latent_with_splitseqs.trainer.rnn_with_splitseqs_trainer_end_recon_only \
    import URLTrainerRnnWithSplitseqsEndReconOnly
from latent_with_splitseqs.trainer.latent_single_layered_full_seq_recon_trainer \
    import URLTrainerLatentWithSplitseqsFullSeqReconLossSingleLayer


df_type_keys = dict(
    feature_extractor='feature_extractor',
    recon='recon',
    rnn_type='rnn_type',
    latent_type='latent_type',
    latent_single_layer_type='latent_single_layer_type'
)

feature_extractor_types = dict(
    rnn='rnn',
    latent_slac='latent_slac',
    latent_single_layer='latent_single_layer'
)

recon_types = dict(
    whole_seq='whole_seq',
    end_only='end_only'
)

rnn_types = dict(
    normal='normal',
    dim_wise='dim_wise',
    normal_posenc='normal_posenc',
    dimwise_posenc='dimwise_posenc',
)

latent_types = dict(
    single_skill='single_skill',
    full_seq='full_seq',
    smoothing='smoothing'
)


def get_df_and_trainer(
        obs_dim,
        seq_len,
        skill_dim,
        df_type,
        df_kwargs_rnn,
        rnn_kwargs,
        df_kwargs_latent,
        latent_kwargs,
        latent_single_layer_kwargs,
        latent_kwargs_smoothing,
        trainer_init_kwargs,
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
        trainer_init_kwargs         : dict, all arguments passed to the trainer init function

    Returns:
        df                          : classifier object
        trainer                     : trainer object

    """
    global df_type_keys
    global feature_extractor_types
    global recon_types
    global rnn_types
    global latent_types

    if df_type[df_type_keys['feature_extractor']] \
        == feature_extractor_types['rnn']:

        # RNN type
        obs_dim_rnn = len(df_kwargs_rnn.obs_dims_used) \
            if df_kwargs_rnn.obs_dims_used is not None \
            else obs_dim
        if df_type[df_type_keys['rnn_type']] == rnn_types['normal']:
            rnn = nn.GRU(
                input_size=obs_dim_rnn,
                hidden_size=rnn_kwargs['hidden_size_rnn'],
                batch_first=True,
                bidirectional=rnn_kwargs['bidirectional'],
            )

        elif df_type[df_type_keys['rnn_type']] == rnn_types['dim_wise']:
            rnn = GRUDimwise(
                input_size=obs_dim_rnn,
                hidden_size=rnn_kwargs['hidden_size_rnn'],
                batch_first=True,
                bidirectional=rnn_kwargs['bidirectional'],
                out_feature_size=rnn_kwargs['hidden_size_rnn']
            )

        elif df_type[df_type_keys['rnn_type']] == rnn_types['normal_posenc']:
            rnn = GRUPosenc(
                input_size=obs_dim_rnn,
                hidden_size=rnn_kwargs['hidden_size_rnn'],
                batch_first=True,
                bidirectional=rnn_kwargs['bidirectional'],
            )

        elif df_type[df_type_keys['rnn_type']] == rnn_types['dimwise_posenc']:
            rnn = GRUDimwisePosenc(
                input_size=obs_dim_rnn,
                hidden_size=rnn_kwargs['hidden_size_rnn'],
                batch_first=True,
                bidirectional=rnn_kwargs['bidirectional'],
                out_feature_size=rnn_kwargs['hidden_size_rnn'] * 2 \
                    if rnn_kwargs['bidirectional'] \
                    else rnn_kwargs['hidden_size_rnn']
            )

        else:
            raise NotImplementedError

        # Classifier using rnn from above
        # Trainer
        if df_type[df_type_keys['recon']] == recon_types['whole_seq']:
            df = SeqwiseSplitseqClassifierRnnWholeSeqRecon(
                seq_len=seq_len,
                obs_dim=obs_dim,
                skill_dim=skill_dim,
                rnn=rnn,
                **df_kwargs_rnn,
            )

            trainer = URLTrainerRnnWithSplitseqsWholeSeqRecon(
                df=df,
                **trainer_init_kwargs
            )

        elif df_type[df_type_keys['recon']] == recon_types['end_only']:
            df = SeqwiseSplitseqClassifierRnnEndReconOnly(
                seq_len=seq_len,
                obs_dim=obs_dim,
                skill_dim=skill_dim,
                rnn=rnn,
                **df_kwargs_rnn,
            )

            trainer = URLTrainerRnnWithSplitseqsEndReconOnly(
                df=df,
                **trainer_init_kwargs
            )

        else:
            raise NotImplementedError


    elif df_type[df_type_keys['feature_extractor']] \
            == feature_extractor_types['latent_slac']:

        # Latent type
        obs_dim_latent = len(df_kwargs_latent.obs_dims_used) \
            if df_kwargs_latent.obs_dims_used is not None \
            else obs_dim
        if df_type[df_type_keys['latent_type']] == latent_types['single_skill']:
            latent_model = SlacLatentNetConditionedOnSingleSkill(
                obs_dim=obs_dim_latent,
                skill_dim=skill_dim,
                **latent_kwargs,
            )

        elif df_type[df_type_keys['latent_type']] == latent_types['full_seq']:
            latent_model = SlacLatentNetConditionedOnSkillSeq(
                obs_dim=obs_dim_latent,
                skill_dim=skill_dim,
                **latent_kwargs,
            )

        elif df_type[df_type_keys['latent_type']] == latent_types['smoothing']:
            latent_model = SlacLatentNetConditionedOnSkillSeqSmoothingPosterior(
                obs_dim=obs_dim_latent,
                skill_dim=skill_dim,
                **latent_kwargs_smoothing,
            )

        else:
            raise  NotImplementedError

        # Classifier using latent model above
        # Trainer
        if df_type[df_type_keys['recon']] == recon_types['whole_seq']:
            df = SeqwiseSplitseqClassifierSlacLatentWholeSeqRecon(
                seq_len=seq_len,
                obs_dim=obs_dim,
                skill_dim=skill_dim,
                latent_net=latent_model,
                **df_kwargs_latent
            )

            trainer = URLTrainerLatentWithSplitseqsFullSeqReconLoss(
                df=df,
                **trainer_init_kwargs
            )

        elif df_type[df_type_keys['recon']] == recon_types['end_only']:
            df = SeqwiseSplitseqClassifierSlacLatentSeqEndOnlyRecon(
                seq_len=seq_len,
                obs_dim=obs_dim,
                skill_dim=skill_dim,
                latent_net=latent_model,
                **df_kwargs_latent
            )

            trainer = URLTrainerLatentWithSplitseqs(
                df=df,
                **trainer_init_kwargs
            )

        else:
            raise NotImplementedError

    elif df_type[df_type_keys['feature_extractor']] \
        == feature_extractor_types['latent_single_layer']:

        # Latent type
        obs_dim_latent = len(df_kwargs_latent.obs_dims_used) \
            if df_kwargs_latent.obs_dims_used is not None \
            else obs_dim

        if df_type[df_type_keys['latent_type']] == latent_types['single_skill']:
            raise NotImplementedError

        elif df_type[df_type_keys['latent_type']] == latent_types['full_seq']:
            latent_model = OneLayeredStochasticLatent(
                obs_dim=obs_dim_latent,
                skill_dim=skill_dim,
                **latent_single_layer_kwargs,
            )

        elif df_type[df_type_keys['latent_type']] == latent_types['smoothing']:
            raise NotImplementedError

        else:
            raise NotImplementedError

        # Classifier using latent model above
        # Trainer
        df = SeqwiseSplitseqClassifierSingleLayeredWholeSeqRecon(
            seq_len=seq_len,
            obs_dim=obs_dim,
            skill_dim=skill_dim,
            latent_net=latent_model,
            **df_kwargs_latent
        )

        trainer = URLTrainerLatentWithSplitseqsFullSeqReconLossSingleLayer(
            df=df,
            **trainer_init_kwargs
        )

    else:
        raise NotImplementedError

    return df, trainer
