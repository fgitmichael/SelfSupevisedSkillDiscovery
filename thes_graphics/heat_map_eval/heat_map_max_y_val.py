def calc_y_max_heat_map(grid_rollout: list):
    coverd_dists = [max(rollout['observations'][:, 1]) for rollout in grid_rollout]
    return coverd_dists
