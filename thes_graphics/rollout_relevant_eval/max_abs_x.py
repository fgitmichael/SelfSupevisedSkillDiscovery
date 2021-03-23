import numpy as np


def extract_max_abs_x(grid_rollout: list, num_to_extract: int) -> list:
    # Calc max abs
    dim = 0
    max_abs_x = [max(abs(rollout['observations'][:, dim])) for rollout in grid_rollout]

    # Sort
    max_abs_x_np = np.array(max_abs_x)
    sort_idx = np.argsort(max_abs_x_np)

    # Extract relevant rollout
    relevant_idx_list = list(sort_idx[-num_to_extract:])
    relevant_grid_rollouts = [grid_rollout[idx] for idx in relevant_idx_list]

    return relevant_grid_rollouts
