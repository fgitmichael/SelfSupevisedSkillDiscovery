import abc


class EvaluationBase(object, metaclass=abc.ABCMeta):

    def __init__(
            self,
            log_prefix: str = None,
            **kwargs
    ):
        if log_prefix is None:
            self.log_prefix = ""
        else:
            self.log_prefix = log_prefix

    @abc.abstractmethod
    def apply_df(self, *args, **kwargs) -> dict:
        """
        Return: df_ret_dict
        """
        raise NotImplementedError

    @abc.abstractmethod
    def classifier_evaluation(self, *args, **kwargs):
        raise NotImplementedError

    def get_log_string(self, _str: str):
        return self.log_prefix + _str
