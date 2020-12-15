from my_utils.dicts.set_config_default_item import set_config_default_item

def prepare_hparams(config):
    config.horizon_len = (config.horizon_len // config.seq_len) * config.seq_len
    config.algorithm_kwargs.batch_size //= config.seq_len

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

    if not "seq_len" in config.df_evaluation_memory.keys():
        config.df_evaluation_memory.seq_len = config.seq_len
    if not "horizon_len" in config.df_evaluation_memory.keys():
        config.df_evaluation_memory.horizon_len = config.horizon_len

    return config
