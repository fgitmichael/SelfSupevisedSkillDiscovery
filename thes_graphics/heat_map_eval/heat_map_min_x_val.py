def calc_x_min_heat_map(grid_rollout: list):
    coverd_dists = [min(rollout['observations'][:, 0]) for rollout in grid_rollout]
    return coverd_dists
