import numpy as np

import rlkit.torch.pytorch_util as ptu


def get_no_oh_grid() -> np.ndarray:
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

    return grid

def get_oh_grid() -> np.ndarray:
    grid = np.eye(10)

    return grid
