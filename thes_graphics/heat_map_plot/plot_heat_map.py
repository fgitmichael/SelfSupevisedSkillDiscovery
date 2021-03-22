import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def plot_heat_map(
        fig: plt.Figure,
        prior_skill_dist: Tuple[tuple, tuple],
        heat_values: np.ndarray,
        log: bool = False,
        width_height: tuple=None,
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

    ## Turn matrix upside down because origin when plotting with pcolor array is turned
    #side_len = heat_values.shape[0]
    #idx = np.arange(side_len)[::-1]
    #heat_values = heat_values[idx]

    ax = fig.add_subplot(1, 1, 1)
    plot = ax.pcolor(heat_values)
    ax.set_xlabel('skill dimension 0')
    ax.set_ylabel('skill dimension 1')
    fig.colorbar(plot)
    _set_fig_size(fig, width_height)

    heat_map_axis, colorbar_axis = fig.get_axes()
    _set_heatmap_axis_ticks_labels(
        heat_map_axis=heat_map_axis,
        prior_skill_dist=prior_skill_dist,
    )
    _set_colorbar_axis_ticks_labels(
        colorbar_axis=colorbar_axis,
        log=log,
    )

def _set_colorbar_axis_ticks_labels(
        colorbar_axis: plt.Axes,
        log,
):
    pass


def _set_heatmap_axis_ticks_labels(
        heat_map_axis: plt.Axes,
        prior_skill_dist: Tuple[tuple, tuple],
):
    heat_map_axis_xticks = heat_map_axis.get_xticks()
    heat_map_axis_yticks = heat_map_axis.get_yticks()
    #assert np.all(heat_map_axis_xticks == heat_map_axis_yticks)
    heat_map_xticks_minmax = np.array([heat_map_axis_xticks[0],
                                       heat_map_axis_xticks[-1]])
    heat_map_yticks_minmax = np.array([heat_map_axis_yticks[0],
                                       heat_map_axis_yticks[-1]])

    x_num_ticks_needed = len(heat_map_axis_xticks)
    y_num_ticks_needed = len(heat_map_axis_yticks)

    heat_map_xticks_new = np.linspace(
        heat_map_xticks_minmax[0],
        heat_map_xticks_minmax[1],
        x_num_ticks_needed,
    )
    heat_map_yticks_new = np.linspace(
        heat_map_yticks_minmax[0],
        heat_map_yticks_minmax[1],
        y_num_ticks_needed,
    )
    heat_map_ylabels_new = np.linspace(
        prior_skill_dist[0][1],
        prior_skill_dist[1][1],
        y_num_ticks_needed,
    )
    heat_map_ylabels_new = ["{:.2f}".format(label) for label in heat_map_ylabels_new]
    heat_map_xlabels_new = np.linspace(
        prior_skill_dist[0][0],
        prior_skill_dist[1][0],
        x_num_ticks_needed,
    )
    heat_map_xlabels_new = ["{:.2f}".format(label) for label in heat_map_xlabels_new]

    heat_map_axis.set_xticks(heat_map_xticks_new)
    heat_map_axis.set_yticks(heat_map_yticks_new)
    heat_map_axis.set_xticklabels(heat_map_xlabels_new)
    heat_map_axis.set_yticklabels(heat_map_ylabels_new)

def _set_fig_size(fig, plot_height_width_inches):
    if plot_height_width_inches is not None:
        fig.set_figheight(plot_height_width_inches[0])
        fig.set_figwidth(plot_height_width_inches[1])
