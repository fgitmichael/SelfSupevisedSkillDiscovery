from my_utils.dicts.set_config_default_item import set_config_default_item
from my_utils.dicts.set_value_if_exist import set_config_value_if_exist

def prepare_hparams(config):
    config.horizon_len = (config.horizon_len // config.seq_len) * config.seq_len
    config.algorithm_kwargs.batch_size //= config.seq_len

    batch_size_latent_key = "batch_size_latent"
    config.algorithm_kwargs = set_config_value_if_exist(
        config=config.algorithm_kwargs,
        key=batch_size_latent_key,
        value=config.algorithm_kwargs[batch_size_latent_key] // config.seq_len,
    )

    config.df_evaluation_env = set_config_default_item(
        config=config.df_evaluation_env,
        key="seq_len",
        default=config.seq_len
    )
    config.df_evaluation_env = set_config_default_item(
        config=config.df_evaluation_env,
        key="horizon_len",
        default=config.horizon_len,
    )

    config.df_evaluation_memory = set_config_default_item(
        config=config.df_evaluation_memory,
        key="seq_len",
        default=config.seq_len,
    )
    config.df_evaluation_memory = set_config_default_item(
        config=config.df_evaluation_memory,
        key="horizon_len",
        default=config.horizon_len,
    )

    return config
