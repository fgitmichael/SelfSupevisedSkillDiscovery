from typing import List, Union
import numpy as np
import abc
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


class PltCreator(object, metaclass=abc.ABCMeta):

    def _plot_line(self,
                  legend_str: str,
                  line_to_plot: np.ndarray):
        if len(line_to_plot.shape) == 2 and line_to_plot.shape[0] == 2:
            plt.plot(line_to_plot[0], line_to_plot[1], label=legend_str)

        else:
            plt.plot(line_to_plot, label=legend_str)

    def _plot_lines(self,
                   legend_str: List[str],
                   arrays_to_plot: Union[List[np.ndarray], np.ndarray]):
        if isinstance(arrays_to_plot, list):
            assert len(legend_str) == len(arrays_to_plot)
        else:
            assert len(legend_str) == arrays_to_plot.shape[0]

        for idx, line in enumerate(arrays_to_plot):
            self._plot_line(legend_str=legend_str[idx],
                           line_to_plot=line)

    def plot_lines(self,
                   legend_str: Union[List[str], str],
                   arrays_to_plot: Union[List[np.ndarray], np.ndarray]) -> plt.Figure:
        if isinstance(arrays_to_plot, list):
            np.stack(arrays_to_plot, dim=0)
            legend_str = list(legend_str)

        self._plot_lines(legend_str=legend_str,
                         arrays_to_plot=arrays_to_plot)

        return plt.gcf()









