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
    import SlacLatentNetConditionedOnSkillSeq, SlacLatentNetConditionedOnSkillSeqForSRNN
from latent_with_splitseqs.latent.\
    slac_latent_conditioned_on_skill_seq_smoothing_posterior \
    import SlacLatentNetConditionedOnSkillSeqSmoothingPosterior
from latent_with_splitseqs.latent.one_stochlayered_latent_conditioned_on_skill_seq \
    import OneLayeredStochasticLatent

from latent_with_splitseqs.latent.df_transformer import DfTransformer
from latent_with_splitseqs.networks.seqwise_splitseq_classifier_transformer \
    import SeqwiseSplitseqClassifierTransformer
from latent_with_splitseqs.trainer.transformer_with_splitseqs_trainer \
    import URLTrainerTransformerWithSplitseqs

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

from latent_with_splitseqs.networks.seqwise_splitseq_classifier_srnn_end_recon_only \
    import SplitSeqClassifierSRNNEndReconOnly
from latent_with_splitseqs.networks.seqwise_splitseq_classifier_srnn_whole_seq_recon \
    import SplitSeqClassifierSRNNWholeSeqRecon
from latent_with_splitseqs.latent.srnn_latent_conditioned_on_skill_seq \
    import SRNNLatentConditionedOnSkillSeq
from latent_with_splitseqs.trainer.latent_srnn_end_recon_only_trainer \
    import URLTrainerLatentSplitSeqsSRNNEndReconOnly
from latent_with_splitseqs.trainer.latent_srnn_full_seq_recon_trainer \
    import URLTrainerLatentSplitSeqsSRNNFullSeqRecon

from latent_with_splitseqs.config.fun.get_obs_dims_used_df import get_obs_dims_used_df


df_type_keys = dict(
    feature_extractor='feature_extractor',
    recon='recon',
    rnn_type='rnn_type',
    latent_type='latent_type',
    latent_single_layer_type='latent_single_layer_type',
    transformer_type='transformer_type',
)

feature_extractor_types = dict(
    rnn='rnn',
    latent_slac='latent_slac',
    latent_single_layer='latent_single_layer',
    srnn='srnn',
    transformer='transformer',
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

transformer_type = dict(
    normal='normal'
)

latent_types = dict(
    single_skill='single_skill',
    full_seq='full_seq',
    smoothing='smoothing'
)


def get_df_and_trainer(
        obs_dim,
        skill_dim,
        df_type,
        df_kwargs_rnn,
        rnn_kwargs,
        df_kwargs_latent,
        latent_kwargs,
        latent_single_layer_kwargs,
        latent_kwargs_smoothing,
        trainer_init_kwargs,
        seq_len=None,
        srnn_kwargs=None,
        df_kwargs_srnn=None,
        df_kwargs_transformer=None,
        transformer_kwargs=None,
        **kwargs
):
    """
    Args:
        obs_dim                     : observation dimensionality
        seq_len                     : sequence length
        skill_dim                   : skill dimensionality
        df_kwargs_rnn               : dict
            obs_dims_used_df        : list or tuple
            dropout                 : float
            hidden_size_rnn         : int
            leaky_slop_classifier   : float
            hidden_units_classifier : list or tuple
        df_type:                    : dict
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
        obs_dims_used_df = get_obs_dims_used_df(
            obs_dim=obs_dim,
            obs_dims_used=df_kwargs_rnn.obs_dims_used,
            obs_dims_used_except=df_kwargs_rnn.obs_dims_used_except \
                if 'obs_dims_used_except' in df_kwargs_rnn else None,
        )
        obs_dim_rnn = len(obs_dims_used_df)
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
        obs_dims_used_df = get_obs_dims_used_df(
            obs_dim=obs_dim,
            obs_dims_used=df_kwargs_latent.obs_dims_used,
            obs_dims_used_except=df_kwargs_latent.obs_dims_used_except
            if 'obs_dims_used_except' in df_kwargs_latent else None,
        )
        obs_dim_latent = len(obs_dims_used_df)
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
            raise NotImplementedError

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

    elif df_type[df_type_keys['feature_extractor']] \
            == feature_extractor_types['srnn']:
        assert srnn_kwargs is not None
        obs_dim_latent = get_obs_dims_used_df(
            obs_dim=obs_dim,
            obs_dims_used=df_kwargs_srnn.obs_dims_used,
            obs_dims_used_except=df_kwargs_srnn.obs_dims_used_except
            if 'obs_dims_used_except' in df_kwargs_srnn else None,
        )
        obs_dim_srnn = len(obs_dim_latent)

        if df_type[df_type_keys['rnn_type']] == rnn_types['normal']:
            rnn = nn.GRU(
                input_size=obs_dim_srnn,
                hidden_size=srnn_kwargs.rnn_kwargs['hidden_size_rnn'],
                batch_first=True,
                bidirectional=srnn_kwargs.rnn_kwargs['bidirectional'],
            )

        else:
            raise NotImplementedError

        if df_type[df_type_keys['latent_type']] == latent_types['full_seq']:
            latent_model_class_stoch = SlacLatentNetConditionedOnSkillSeqForSRNN
            srnn_model = SRNNLatentConditionedOnSkillSeq(
                obs_dim=obs_dim_srnn,
                skill_dim=skill_dim,
                filter_net_params=srnn_kwargs.filter_net_params,
                deterministic_latent_net=rnn,
                stochastic_latent_net_class=latent_model_class_stoch,
                stochastic_latent_net_class_params=srnn_kwargs.stoch_latent_kwargs,
            )

        else:
            raise NotImplementedError

        if df_type[df_type_keys['recon']] == recon_types['end_only']:
            df = SplitSeqClassifierSRNNEndReconOnly(
                obs_dim=obs_dim,
                skill_dim=skill_dim,
                seq_len=seq_len,
                latent_net=srnn_model,
                **df_kwargs_srnn
            )
            trainer = URLTrainerLatentSplitSeqsSRNNEndReconOnly(
                df=df,
                **trainer_init_kwargs
            )

        elif df_type[df_type_keys['recon']] == recon_types['whole_seq']:
            df = SplitSeqClassifierSRNNWholeSeqRecon(
                obs_dim=obs_dim,
                skill_dim=skill_dim,
                seq_len=seq_len,
                latent_net=srnn_model,
                **df_kwargs_srnn
            )
            trainer = URLTrainerLatentSplitSeqsSRNNFullSeqRecon(
                df=df,
                **trainer_init_kwargs
            )

        else:
            raise NotImplementedError

    elif df_type[df_type_keys['feature_extractor']]\
            == feature_extractor_types['transformer']:
        obs_dims_used_df = get_obs_dims_used_df(
            obs_dim=obs_dim,
            obs_dims_used=df_kwargs_transformer.obs_dims_used,
            obs_dims_used_except=df_kwargs_transformer.obs_dims_used_except
            if 'obs_dims_used_except' in df_kwargs_transformer else None,
        )
        num_obs_dims_used = len(obs_dims_used_df)

        if df_type[df_type_keys['transformer_type']] == transformer_type['normal']:
            transformer = DfTransformer(
                input_size=num_obs_dims_used,
                **transformer_kwargs,
            )

        else:
            raise NotImplementedError

        assert seq_len is not None
        df = SeqwiseSplitseqClassifierTransformer(
            seq_len=seq_len,
            obs_dim=obs_dim,
            skill_dim=skill_dim,
            df_transformer=transformer,
            **df_kwargs_transformer
        )

        trainer = URLTrainerTransformerWithSplitseqs(
            df=df,
            **trainer_init_kwargs
        )

    else:
        raise NotImplementedError

    return df, trainer
