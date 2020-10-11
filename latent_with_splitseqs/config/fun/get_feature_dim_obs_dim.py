from latent_with_splitseqs.config.fun.get_df import df_type_keys, feature_extractor_types

def get_feature_dim_obs_dim(
        obs_dim,
        config
):
    if config.df_type[df_type_keys['feature_extractor']] == feature_extractor_types['rnn']:
        feature_dim_or_obs_dim = config.df_kwargs.hidden_size_rnn \
            if config['trainer_kwargs']['train_sac_in_feature_space'] \
            else obs_dim

    elif config.df_type[df_type_keys['feature_extractor']] \
        == feature_extractor_types['latent']:
        raise NotImplementedError

    else:
        raise NotImplementedError

    return feature_dim_or_obs_dim
