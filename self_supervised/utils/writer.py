import os
from datetime import datetime
from typing import Union, List

import numpy as np
import torch
from prodict import Prodict
from torch.utils.tensorboard import SummaryWriter

from self_supervised.base.writer.plt_creator_base import PltCreator


class WriterDataMapping(Prodict):
    value: torch.Tensor
    global_step: int

    def __init__(self,
                 value: torch.Tensor,
                 global_step: int):
        super().__init__(
            value=value,
            global_step=global_step
        )


class MyWriter(object):

    def __init__(self,
                 seed: int,
                 log_dir: str,
                 run_comment=None):
        self.log_dir = log_dir

        run_id = f'mode_disent{seed}-{datetime.now().strftime("%Y%m%d-%H%M")}'
        run_id += run_comment if type(run_comment) is str else ""

        self.model_dir = os.path.join(self.log_dir, str(run_id), 'model')
        self.summary_dir = os.path.join(self.log_dir, str(run_id), 'summary')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(
            log_dir=self.summary_dir,
        )
        self.plt_creator = PltCreator()

    def plot(self,
             *args,
             tb_str: str,
             step: int,
             labels: Union[List[str], str] = None,
             x_lim = None,
             y_lim = None,
             **kwargs
             ):
        fig = self.plt_creator.plot(
            *args,
            labels = labels,
            x_lim = x_lim,
            y_lim = y_lim,
            **kwargs)

        self.writer.add_figure(
            tag=tb_str,
            figure=fig,
            global_step=step,
        )

    def plot_lines(self,
                   legend_str: Union[str, List[str]],
                   tb_str: str,
                   arrays_to_plot: Union[np.ndarray, List[np.ndarray]],
                   step: int,
                   x_lim = None,
                   y_lim = None):

        fig = self.plt_creator.plot_lines(
            legend_str=legend_str,
            arrays_to_plot=arrays_to_plot,
            x_lim=x_lim,
            y_lim=y_lim
        )

        self.writer.add_figure(
            tag=tb_str,
            figure=fig,
            global_step=step
        )

    def save_models(self,
                    models: dict):
        for k, v in models.items():
            path_name = os.path.join(self.model_dir, k + '.pkl')
            torch.save(v, path_name)

    def log_dict_scalars(self,
                         dict_to_log: dict,
                         step: int,
                         base_tag: str = None):
        if base_tag is None:
            base_tag = ''

        for k, v in dict_to_log.items():
            self.writer.add_scalar(
                tag="{}/{}".format(base_tag, k),
                scalar_value=v,
                global_step=step
            )