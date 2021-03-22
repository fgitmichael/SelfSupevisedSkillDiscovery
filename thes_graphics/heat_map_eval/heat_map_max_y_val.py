def calc_covered_dist_heat_map(grid_rollout: list):
    coverd_dists = [max(rollout['observations'][:, 0]) for rollout in grid_rollout]
    return coverd_dists
