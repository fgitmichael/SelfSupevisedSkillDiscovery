from mode_disent_no_ssm.utils.parse_args import parse_args

import self_sup_combined.utils.typed_dicts as tdssc


def parse_variant() -> tdssc.VariantMapping:
    config_args = parse_args()

    variant = tdssc.VariantMapping(
        env_kwargs=config_args.EnvKwargs,
        algo_kwargs=config_args.AlgoKwargs,
        trainer_kwargs=config_args.TrainerKwargs,
        mode_encoder_kwargs=config_args.ModeEncoderKwargs,
        info_loss_kwargs=config_args.InfoLossKwargs,
        **config_args.Other
    )

    return variant
