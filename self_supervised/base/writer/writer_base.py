from typing import Union, List
import numpy as np
import torch
import abc


class WriterBase(object):


    @abc.abstractmethod
    def plot_lines(self,
                   legend_str,
                   tb_str,
                   arrays_to_plot,
                   step,
                   x_lim = None,
                   y_lim = None):
        raise NotImplementedError

    @abc.abstractmethod
    def save_models(self,
                    models):
        raise NotImplementedError

    @abc.abstractmethod
    def log_dict_scalars(self,
                         dict_to_log ,
                         step,
                         base_tag):
        raise NotImplementedError
