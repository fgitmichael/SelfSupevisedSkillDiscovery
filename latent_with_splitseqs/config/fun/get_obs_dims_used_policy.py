def get_obs_dims_used_policy(
        obs_dim,
        config
):
    """
    Args:
        obs_dim                 : int
        config                  : containing key obs_dims_used_policy
    Returns:
        obs_dims_used_policy    : tuple
    """
    key = 'obs_dims_used_policy'
    default = [i for i in range(obs_dim)]
    if key in config.keys():
        if config[key] is not None:
            assert isinstance(config[key], list) or \
                   isinstance(config[key], tuple)
            assert  config.trainer_kwargs.train_sac_in_feature_space is False
            obs_dims_used_policy = config[key]

        else:
            obs_dims_used_policy = default

    else:
        obs_dims_used_policy = default

    return obs_dims_used_policy
