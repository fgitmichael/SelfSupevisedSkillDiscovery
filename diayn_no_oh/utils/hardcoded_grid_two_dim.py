import numpy as np

import rlkit.torch.pytorch_util as ptu

class NoohGridCreator(object):
    def __init__(self,
                 radius_factor=1,
                 repeat=1):
        self.radius_factor = radius_factor
        self.repeat = repeat

    def get_grid(self) -> np.ndarray:
        # Hard coded for testing
        radius_factor = 1
        repeat = 10

        radius1 = 0.75 * radius_factor
        radius2 = 1. * radius_factor
        radius3 = 1.38 * radius_factor
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

        # Repeat
        grid = np.concatenate([grid] * repeat, axis=-1)

        return grid

class OhGridCreator(object):
    def get_grid(self) -> np.ndarray:
        grid = np.eye(10)

        return grid
