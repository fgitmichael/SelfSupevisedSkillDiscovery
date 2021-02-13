from my_utils.dicts.set_config_default_item import set_config_default_item
from my_utils.dicts.set_value_if_exist import set_config_value_if_exist
from my_utils.dicts.get_config_item import get_config_item

def prepare_hparams(config):

    replay_seq_sampling_variant = get_config_item(
        config=config,
        key='replay_seq_sampling',
    )
    if replay_seq_sampling_variant == 'sampling_random_seq_len':
        max_seq_len = config.max_sample_seq_len
        seq_len = max_seq_len

    else:
        seq_len = config.seq_len

    config.horizon_len = (config.horizon_len // seq_len) * seq_len
    config.algorithm_kwargs.batch_size //= seq_len

    # Batch size latent
    batch_size_latent_key = "batch_size_latent"
    config.algorithm_kwargs = set_config_value_if_exist(
        config=config.algorithm_kwargs,
        key=batch_size_latent_key,
        value=config.algorithm_kwargs[batch_size_latent_key] // seq_len,
    )

    # Evaluation seq len setting
    config.df_evaluation_env = set_config_default_item(
        config=config.df_evaluation_env,
        key="seq_len",
        default=seq_len
    )

    config.df_evaluation_memory = set_config_default_item(
        config=config.df_evaluation_memory,
        key="seq_len",
        default=seq_len,
    )

    # Set horizon len for env evaluation
    config.df_evaluation_env = set_config_default_item(
        config=config.df_evaluation_env,
        key="horizon_len",
        default=config.horizon_len,
    )

    # Set horizon len for memory evaluation
    config.df_evaluation_memory = set_config_default_item(
        config=config.df_evaluation_memory,
        key="horizon_len",
        default=config.horizon_len,
    )

    return config
