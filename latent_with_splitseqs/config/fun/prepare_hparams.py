def prepare_hparams(config):
    config.horizon_len = (config.horizon_len // config.seq_len) * config.seq_len
    config.algorithm_kwargs.batch_size //= config.seq_len

    if not "seq_len" in config.df_evaluation_env.keys():
        config.df_evaluation_env.seq_len = config.seq_len
    if not "horizon_len" in config.df_evaluation_env.keys():
        config.df_evaluation_env.horizon_len = config.horizon_len

    if not "seq_len" in config.df_evaluation_memory.keys():
        config.df_evaluation_memory.seq_len = config.seq_len
    if not "horizon_len" in config.df_evaluation_memory.keys():
        config.df_evaluation_memory.horizon_len = config.horizon_len

    return config
