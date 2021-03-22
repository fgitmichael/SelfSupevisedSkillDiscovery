import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def plot_heat_map(
        fig: plt.Figure,
        prior_skill_dist: Tuple[tuple, tuple],
        heat_values: np.ndarray,
        log: bool = False,
):
    """
    Args:
        prior_skill_dist        : ((x,y), (x,y)) tuple with to two dimensional coordinates
        heat_values             : (sqrt(N), sqrt(N)) numpy array where N is the
                                  number of values
    """
    assert heat_values.shape[0] == heat_values.shape[1]
    assert len(prior_skill_dist) == 2
    assert len(prior_skill_dist[0]) == len(prior_skill_dist[1]) == 2

    if log:
        heat_values = np.log(heat_values)

    ax = fig.add_subplot(1, 1, 1)
    plot = ax.pcolor(heat_values)
    fig.colorbar(plot)
