import abc


class LossFunBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def loss(self,
             pri,
             post,
             recon,
             data,
             **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def input_assertions(self,
                         pri,
                         post,
                         recon,
                         data,
                         ):
        raise NotImplementedError
