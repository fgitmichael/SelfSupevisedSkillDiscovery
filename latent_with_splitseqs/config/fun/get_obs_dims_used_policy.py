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
    key = 'obs_dims_used_policy_all_except'

    default = [i for i in range(obs_dim)]
    obs_dims_used_policy = default
    if key in config.keys():
        if config[key] is not None:
            assert isinstance(config[key], list) or \
                   isinstance(config[key], tuple)
            obs_dims_used_policy = [i for i in default if i not in config[key]]

    return obs_dims_used_policy
