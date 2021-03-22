import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def plot_heat_map(
        prior_skill_dist: Tuple[tuple, tuple],
        heat_values: np.ndarray,
        log: bool = False,
) -> plt.Figure:
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

    fig = plt.figure()
    plt.imshow(heat_values)
    plt.colorbar(orientation='vertical')

    heat_map_axis, colorbar_axis = fig.get_axes()
    set_heatmap_axis_ticks_labels(
        heat_map_axis=heat_map_axis,
        num_ticks_needed=heat_values.shape[0] + 1,
        prior_skill_dist=prior_skill_dist,
    )
    set_colorbar_axis_ticks_labels(
        colorbar_axis=colorbar_axis,
        log=log,
    )

    return fig


def set_colorbar_axis_ticks_labels(
        colorbar_axis: plt.Axes,
        log,
):
    pass


def set_heatmap_axis_ticks_labels(
        heat_map_axis: plt.Axes,
        num_ticks_needed: int,
        prior_skill_dist: Tuple[tuple, tuple],
):
    heat_map_axis_xticks = heat_map_axis.get_xticks()
    heat_map_axis_yticks = heat_map_axis.get_yticks()
    assert np.all(heat_map_axis_xticks == heat_map_axis_yticks)
    heat_map_xticks_minmax = np.array([heat_map_axis_xticks[0],
                                       heat_map_axis_xticks[-1]])
    heat_map_yticks_minmax = np.array([heat_map_axis_yticks[0],
                                       heat_map_axis_yticks[-1]])

    heat_map_xticks_new = np.linspace(
        heat_map_xticks_minmax[0],
        heat_map_xticks_minmax[1],
        num_ticks_needed,
    )
    heat_map_yticks_new = np.linspace(
        heat_map_yticks_minmax[0],
        heat_map_yticks_minmax[1],
        num_ticks_needed,
    )
    heat_map_ylabels_new = np.linspace(
        prior_skill_dist[0][1],
        prior_skill_dist[1][1],
        num_ticks_needed,
    )
    heat_map_ylabels_new = ["{:.2f}".format(label) for label in heat_map_ylabels_new]
    heat_map_xlabels_new = np.linspace(
        prior_skill_dist[0][0],
        prior_skill_dist[1][0],
        num_ticks_needed,
    )
    heat_map_xlabels_new = ["{:.2f}".format(label) for label in heat_map_xlabels_new]

    heat_map_axis.set_xticks(heat_map_xticks_new)
    heat_map_axis.set_yticks(heat_map_yticks_new)
    heat_map_axis.set_xticklabels(heat_map_xlabels_new)
    heat_map_axis.set_yticklabels(heat_map_ylabels_new)
