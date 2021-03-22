def calc_covered_dist_heat_map(grid_rollout: list):
    coverd_dists = [rollout['observations'][-1, 0] for rollout in grid_rollout]
    return coverd_dists
