def get_obs_dims_used_df(
        obs_dim,
        obs_dims_used=None,
        obs_dims_used_except=None,
):
    obs_dims_used_default = [i for i in range(obs_dim)]
    if obs_dims_used is not None and obs_dims_used_except is None:
        assert isinstance(obs_dims_used, list) \
               or isinstance(obs_dims_used, tuple)
        used_dims = obs_dims_used

    elif obs_dims_used is None and obs_dims_used_except is not None:
        used_dims = [i for i in obs_dims_used_default
                     if i not in obs_dims_used_except]

    elif obs_dims_used is None and obs_dims_used_except is None:
        used_dims = obs_dims_used_default

    else:
        raise NotImplementedError

    return used_dims
