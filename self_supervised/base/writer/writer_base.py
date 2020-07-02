from typing import Union, List
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from self_supervised.base.writer.plt_creator_base import PltCreator


class WriterBase(object):

    def __init__(self,
                 writer: SummaryWriter,
                 plt_creator: PltCreator):
        self.writer = writer
        self.plt_creator = plt_creator

    def add_scalar(self,
                   str: str,
                   value,
                   step: int):
        self.writer.add_scalar(
            scalar_value=value,
            tag=str,
            global_step=step,
        )

    def plot_lines(self,
                   legend_str: Union[str, List[str]],
                   tb_str: str,
                   arrays_to_plot: Union[np.ndarray, List[np.ndarray]],
                   step: int):

        fig = self.plt_creator.plot_lines(
            legend_str=legend_str,
            arrays_to_plot=arrays_to_plot)

        self.writer.add_figure(
            tag=tb_str,
            figure=fig,
            global_step=step
        )





