import math

import numpy as np


def create_twod_grid(
        low: np.ndarray,
        high: np.ndarray,
        num_points: int,
        matrix_form=False,
        **kwargs
) -> np.ndarray:
    """
    Creates grid with n=ceil(sqrt(num_points) points
    low             : (d,) np.array
    high            : (d,) np.array
    num_points      : N
    matrix_form     : bool
    """
    assert low.shape == high.shape
    assert len(low.shape) == 1

    num_points_array = math.ceil(math.sqrt(num_points))
    mesh_grid_arrays = []
    for dim_low, dim_high in zip(low, high):
        array_ = np.linspace(dim_low, dim_high, num_points_array)
        mesh_grid_arrays.append(array_)

    grid_list = np.meshgrid(*mesh_grid_arrays)
    assert len(grid_list) == low.shape[0]
    assert isinstance(grid_list, list)

    grid = np.stack(grid_list, axis=-1)
    if matrix_form:
        grid = np.reshape(grid, (num_points_array * num_points_array, low.shape[-1]))

    return grid