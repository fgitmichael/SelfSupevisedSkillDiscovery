import os
from datetime import datetime
from typing import Union, List, Tuple
from latent_with_splitseqs.base.my_object_base import MyObjectBase
import numpy as np
import torch
from prodict import Prodict
from torch.utils.tensorboard import SummaryWriter
import shutil

from self_supervised.base.writer.plt_creator_base import PltCreator
from self_supervised.base.writer.writer_base import WriterBase


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


class MyWriter(WriterBase, MyObjectBase):

    def __init__(self,
                 seed: int,
                 log_dir: str,
                 run_comment=None):

        run_id = f'mode_disent{seed}-{datetime.now().strftime("%Y%m%d-%H%M")}'
        run_id += run_comment if type(run_comment) is str else ""

        self.run_dir = self.get_run_dir_name(
            log_dir=log_dir,
            run_id=run_id,
        )

        self.model_dir = ''
        self.summary_dir = ''
        self.writer = None
        self.set_up_directory_and_create_summary_writer(run_dir=self.run_dir)

        self.plt_creator = PltCreator()

    def set_up_directory_and_create_summary_writer(self, run_dir):
        self.model_dir = os.path.join(run_dir, 'model')
        self.summary_dir = os.path.join(run_dir, 'summary')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(
            log_dir=self.summary_dir,
        )

    def create_save_dict(self) -> dict:
        save_obj = super().create_save_dict()
        save_obj['run_dir'] = self.run_dir
        return save_obj

    def load(
            self,
            file_name,
            base_dir='.',
            delete_current_run_dir=False
    ):
        save_path = self._get_save_path(
            file_name=file_name,
            base_dir=base_dir,
        )
        save_obj = torch.load(save_path)
        self.process_save_dict(save_obj, delete_current_run_dir=delete_current_run_dir)

    def process_save_dict(
            self,
            save_obj,
            delete_current_run_dir=None
    ):
        if delete_current_run_dir is None:
            raise ValueError('arg delete_current_run_dir is required')

        old_run_dir = self.run_dir
        self.run_dir = save_obj['run_dir']
        self.set_up_directory_and_create_summary_writer(run_dir=self.run_dir)
        if delete_current_run_dir:
            shutil.rmtree(old_run_dir)

        super().process_save_dict(save_obj)

    def __del__(self):
        pass
        #print("Close Writer")
        #self.writer.close()

    def get_run_dir_name(self, run_id, log_dir):
        run_id_try = run_id
        cnt = 0
        while os.path.exists(os.path.join(log_dir, str(run_id_try))):
            cnt += 1
            run_id_try = run_id + "_try_" + str(cnt)

        return os.path.join(log_dir, str(run_id_try))

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

    def scatter(self,
                *args,
                tb_str: str,
                step: int,
                labels: Union[List[str], str]=None,
                x_lim=None,
                y_lim=None,
                **kwargs
                ):
        fig = self.plt_creator.scatter(
            *args,
            labels=labels,
            x_lim=x_lim,
            y_lim=y_lim,
            **kwargs
        )

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


class EmptyWriter(SummaryWriter):

    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_figure(self, *args, **kwargs):
        pass


class MyWriterWithActivation(MyWriter):

    def __init__(self,
                 *args,
                 activate=True,
                 **kwargs):
        self.activate = activate

        if self.activate:
            super().__init__(*args, **kwargs)

        else:
            self.writer = EmptyWriter()

    def plot(self,
             *args,
             tb_str: str,
             step: int,
             labels: Union[List[str], str] = None,
             x_lim = None,
             y_lim = None,
             **kwargs
             ):
        if self.activate:
            super().plot(*args,
                         tb_str=tb_str,
                         step=step,
                         labels=labels,
                         x_lim=x_lim,
                         y_lim=y_lim,
                         **kwargs)

    def plot_lines(self,
                   legend_str: Union[str, List[str]],
                   tb_str: str,
                   arrays_to_plot: Union[np.ndarray, List[np.ndarray]],
                   step: int,
                   x_lim = None,
                   y_lim = None):
        if self.activate:
            super().plot_lines(
                legend_str=legend_str,
                tb_str=tb_str,
                arrays_to_plot=arrays_to_plot,
                step=step,
                x_lim=x_lim,
                y_lim=y_lim
            )

    def save_models(self,
                    models: dict):
        if self.activate:
            super().save_models(
                models=models
            )

    def log_dict_scalars(self,
                         dict_to_log: dict,
                         step: int,
                         base_tag: str = None):
        if self.activate:
            super().log_dict_scalars(
                dict_to_log=dict_to_log,
                step=step,
                base_tag=base_tag
            )
