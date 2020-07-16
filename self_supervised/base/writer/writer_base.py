import torch
import os
from typing import Union, List
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from prodict import Prodict
from datetime import datetime

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


class WriterBase(object):

    def __init__(self,
                 seed: int,
                 log_dir: str,
                 plt_creator: PltCreator):
        self.log_dir = log_dir

        run_id = f'mode_disent{seed}-{datetime.now().strftime("%Y%m%d-%H%M")}'
        self.model_dir = os.path.join(self.log_dir, str(run_id), 'model')
        self.summary_dir = os.path.join(self.log_dir, str(run_id), 'summary')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.plt_creator = plt_creator

    def add_scalar(self,
                   tag: str,
                   scalar_mapping: WriterDataMapping):
        self.writer.add_scalar(
            tag=tag,
            **scalar_mapping
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

    def save_models(self,
                    models: dict):
        for k, v in models.items():
            path_name = os.path.join(self.model_dir, k + '.pkl')
            torch.save(v, path_name)
