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

        plt.legend()

    def _plot_lines(self,
                    legend_str: Union[List[str], str],
                   arrays_to_plot: Union[List[np.ndarray], np.ndarray]):
        if isinstance(arrays_to_plot, list):
            assert len(legend_str) == len(arrays_to_plot)

        elif len(arrays_to_plot.shape) > 1:
            assert len(legend_str) == arrays_to_plot.shape[0]

        else:
            assert type(legend_str) is str
            legend_str = [legend_str]
            arrays_to_plot = np.expand_dims(arrays_to_plot, axis=0)


        for idx, line in enumerate(arrays_to_plot):
            self._plot_line(
                legend_str=legend_str[idx],
                line_to_plot=line
            )

    def plot_lines(self,
                   legend_str: Union[List[str], str],
                   arrays_to_plot: Union[List[np.ndarray], np.ndarray],
                   x_lim=None,
                   y_lim=None) -> plt.Figure:
        if isinstance(arrays_to_plot, list):
            np.stack(arrays_to_plot, axis=0)
            legend_str = list(legend_str)

        plt.clf()
        self._plot_lines(legend_str=legend_str,
                         arrays_to_plot=arrays_to_plot)

        ax = plt.gca()
        if x_lim is not None:
            ax.set_xlim(x_lim)

        if y_lim is not None:
            ax.set_ylim(y_lim)


        return plt.gcf()

    def plot(self,
             *args,
             labels,
             x_lim=None,
             y_lim=None,
             **kwargs):
        lineobjects = plt.plot(*args, **kwargs)

        if labels is not None:
            plt.legend(lineobjects, labels)

        if x_lim is not None:
            plt.xlim(*x_lim)
        if y_lim is not None:
            plt.ylim(*y_lim)

        return plt.gcf()

    def scatter(self,
                *args,
                labels,
                x_lim=None,
                y_lim=None,
                **kwargs):
        """
        Args:
            *args           : data
            ...
            **kwargs        : i.e. color
        """
        lineobjects = plt.plot(*args, **kwargs)

        if labels is not None:
            plt.legend(lineobjects, labels)

        if x_lim is not None:
            plt.xlim(*x_lim)
        if y_lim is not None:
            plt.ylim(*y_lim)

        return plt.gcf()

