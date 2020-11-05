from functools import wraps
from typing import Type

from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb


def add_post_epoch_func(post_epoch_func):
    def wrap_class(
            algo_class: Type[DIAYNTorchOnlineRLAlgorithmTb],
    ):
        class AlgoClassCopy(algo_class):
            pass

        orig_init = AlgoClassCopy.__init__

        @wraps(orig_init)
        def new_init(self,
                     *args,
                     **kwargs):
            orig_init(self, *args, **kwargs)
            self.post_epoch_funcs.append(post_epoch_func)

        AlgoClassCopy.__init__ = new_init

        return AlgoClassCopy
    return wrap_class
