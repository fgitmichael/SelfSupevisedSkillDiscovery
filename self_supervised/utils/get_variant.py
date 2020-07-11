from mode_disent_no_ssm.utils.parse_args import parse_args

import self_supervised.utils.typed_dicts as td


def parse_variant() -> td.VariantMapping:
    config_args = parse_args()

    variant = td.VariantMapping(
        env_kwargs=config_args.EnvKwargs,
        algo_kwargs=config_args.AlgoKwargs,
        trainer_kwargs=config_args.TrainerKwargs,
        mode_latent_kwargs=config_args.ModeLatentKwargs,
        info_loss_kwargs=config_args.InfoLossKwargs,
        **config_args.Other
    )

    return variant
