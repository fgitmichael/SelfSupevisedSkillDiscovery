def calc_covered_y_heat_map(grid_rollout: list):
    coverd_dists = [rollout['observations'][-1, 1] for rollout in grid_rollout]
    return coverd_dists
