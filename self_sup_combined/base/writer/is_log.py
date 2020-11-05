from functools import wraps

from latent_with_splitseqs.algo.algo_latent_splitseqs import SeqwiseAlgoRevisedSplitSeqs


def is_log(log_interval=None):
    def check_is_log(func):
        """
        Decorate to wrap is_log method around other functions
        """
        @wraps(func)
        def new_fun(
                self,
                *args,
                **kwargs
        ):
            assert 'epoch' in kwargs.keys()
            assert isinstance(self, SeqwiseAlgoRevisedSplitSeqs)

            epoch = kwargs['epoch']
            if self.diagnostic_writer.is_log(epoch, log_interval=log_interval):
                return func(*args, **kwargs)
        return new_fun
    return check_is_log