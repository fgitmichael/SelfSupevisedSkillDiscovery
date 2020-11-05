import abc
from typing import Union

from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter


class EnvEvaluationBase(object, metaclass=abc.ABCMeta):

    def __init__(
            self,
            seq_collector,
            df_to_evaluate,
            obs_dims_to_log: Union[list, tuple],
            diagnostics_writer: DiagnosticsWriter,
            plot_skill_influence: dict = None,
            action_dims_to_log: Union[list, tuple] = None,
    ):
        self.seq_collector = seq_collector
        self.diagno_writer = diagnostics_writer
        self.obs_dims_to_log = obs_dims_to_log
        self.df_to_evaluate = df_to_evaluate

        if plot_skill_influence is None:
            self.plot_skill_influence = dict(
                obs=False,
                action=False,
                obs_one_plot=False,
                plot_post=False,
            )
        else:
            self.plot_skill_influence = plot_skill_influence

        if action_dims_to_log is None:
            self.action_dims_to_log = [i for i in range(self.seq_collector.action_dim)]
        else:
            self.action_dims_to_log = action_dims_to_log

    def __call__(self, epoch):
        eval_paths_dict = self.collect_skill_influence_paths()

        self.plot_mode_influence_paths(
            epoch=epoch,
            **eval_paths_dict
        )

        df_ret_dict = self.apply_df(**eval_paths_dict)

        self.plot_posterior(
            epoch=epoch,
            **df_ret_dict,
            **eval_paths_dict,
        )

        self.classifier_evaluation(
            epoch=epoch,
            **df_ret_dict,
            **eval_paths_dict
        )

    @abc.abstractmethod
    def collect_skill_influence_paths(self) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def plot_mode_influence_paths(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def apply_df(self, *args, **kwargs) -> dict:
        """
        Return: df_ret_dict
        """
        raise NotImplementedError

    @abc.abstractmethod
    def plot_posterior(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def classifier_evaluation(self, *args, **kwargs):
        raise NotImplementedError
