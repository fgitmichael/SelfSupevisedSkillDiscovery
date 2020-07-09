from mode_disent_no_ssm.utils.parse_args import parse_args

from self_supervised.utils.typed_dicts import *


def parse_variant() -> VariantMapping:
    config_args = parse_args()

    variant = VariantMapping(
        env_kwargs=config_args.EnvKwargs,
        algo_kwargs=config_args.AlgoKwargs,
        trainer_kwargs=config_args.TrainerKwargs,
        mode_latent_kwargs=config_args.ModeLatentKwargs,
        info_loss_kwargs=config_args.InfoLossKwargs,
        **config_args.Other
    )

    return variant





