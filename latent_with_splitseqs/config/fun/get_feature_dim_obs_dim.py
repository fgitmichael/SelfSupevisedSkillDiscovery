from latent_with_splitseqs.config.fun.get_df_and_trainer import \
    df_type_keys, feature_extractor_types, latent_types

def get_feature_dim_obs_dim(
        obs_dim,
        config
):
    if config.df_type[df_type_keys['feature_extractor']] == feature_extractor_types['rnn']:
        hidden_size = 2 * config.rnn_kwargs.hidden_size_rnn \
            if config.rnn_kwargs.bidirectional \
            else config.rnn_kwargs.hidden_size_rnn
        feature_dim_or_obs_dim = hidden_size \
            if config['trainer_kwargs']['train_sac_in_feature_space'] \
            else obs_dim

    elif config.df_type[df_type_keys['feature_extractor']] \
        == feature_extractor_types['latent_slac']:
        latent_dim = config.latent_kwargs.latent1_dim + \
                     config.latent_kwargs.latent2_dim \
            if config.df_type.latent_type is not latent_types['smoothing'] \
            else config.latent_kwargs_smoothing.latent1_dim + \
                 config.latent_kwargs_smoothing.latent1_dim
        feature_dim_or_obs_dim = latent_dim \
            if config['trainer_kwargs']['train_sac_in_feature_space'] \
            else obs_dim

    elif config.df_type[df_type_keys['feature_extractor']] \
        == feature_extractor_types['latent_single_layer']:
        latent_dim = config.latent_single_layer_kwargs.latent_dim \
            if config.df_type.latent_type is not latent_types['smoothing'] \
            else config.df_type.latent_kwargs_smoothing.latent_dim
        feature_dim_or_obs_dim = latent_dim \
            if config['trainer_kwargs']['train_sac_in_feature_space'] \
            else obs_dim

    else:
        raise NotImplementedError

    return feature_dim_or_obs_dim
