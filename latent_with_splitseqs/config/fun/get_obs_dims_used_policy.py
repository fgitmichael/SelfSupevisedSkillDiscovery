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
    if key in config.keys():
        assert isinstance(config[key], list) or \
               isinstance(config[key], tuple)
        return config[key]

    else:
        return [i for i in range(obs_dim)]
