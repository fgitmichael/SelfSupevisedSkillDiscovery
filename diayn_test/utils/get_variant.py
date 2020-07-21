from mode_disent_no_ssm.utils.parse_args import parse_args

import diayn_test.utils.typed_dicts as tddt

def parse_variant() -> tddt.VariantMapping:
    config_args = parse_args()

    variant = tddt.VariantMapping(
        env_kwargs=config_args.EnvKwargs,
        algo_kwargs=config_args.AlgoKwargs,
        trainer_kwargs=config_args.TrainerKwargs,
        **config_args.Other
    )

    return variant