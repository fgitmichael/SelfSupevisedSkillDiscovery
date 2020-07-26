import numpy as np


def get_grid():
    # Hard coded for testing
    radius1 = 0.75
    radius2 = 1.
    radius3 = 1.38
    grid = np.array([
        [0., 0.],
        [radius1, 0.],
        [0., radius1],
        [-radius1, 0.],
        [0, -radius1],
        [radius2, radius2],
        [-radius2, radius2],
        [radius2, -radius2],
        [-radius2, -radius2],
        [0, radius3]
    ], dtype=np.float)

    grid = ptu.from_numpy(grid)

    return grid
