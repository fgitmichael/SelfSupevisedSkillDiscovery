import numpy as np


def extract_max_plus_minus_x(grid_rollout: list, num_to_extract: int) -> list:
    num_to_extract = num_to_extract if num_to_extract % 2 == 0 else num_to_extract + 1

    # Calc max abs
    dim = 0
    max_x = [max(rollout['observations'][:, dim]) for rollout in grid_rollout]
    min_x = [min(rollout['observations'][:, dim]) for rollout in grid_rollout]

    # Sort
    max_x_np = np.array(max_x)
    min_x_np = np.array(min_x)
    sort_idx_max_x = np.argsort(max_x_np)
    sort_idx_min_x = np.argsort(min_x_np)

    # Extract relevant rollouts
    relevant_idx_list_plus = list(sort_idx_max_x[-num_to_extract//2:])
    relevant_idx_list_minus = list(sort_idx_min_x[:num_to_extract//2])
    relevant_idx_list = []
    relevant_idx_list.extend(relevant_idx_list_plus)
    relevant_idx_list.extend(relevant_idx_list_minus)
    relevant_grid_rollouts = [grid_rollout[idx] for idx in relevant_idx_list]

    return relevant_grid_rollouts

