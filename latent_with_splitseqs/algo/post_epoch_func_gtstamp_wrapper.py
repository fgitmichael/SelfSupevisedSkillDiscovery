import gtimer as gt

from functools import wraps

from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb


def post_epoch_func_wrapper(
        gt_stamp_name,
        log_interval=None,
        method: bool=False,
):
    def real_dec(func):
        @wraps(func)
        def new_func(self, *args, **kwargs):
            epoch = kwargs['epoch']
            assert isinstance(self, DIAYNTorchOnlineRLAlgorithmTb)

            if self.diagnostic_writer.is_log(epoch, log_interval=log_interval):
                if method:
                    ret = func(self, *args, **kwargs)
                else:
                    ret = func(*args, **kwargs)

            else:
                ret = None
            gt.stamp(gt_stamp_name)

            return ret

        return new_func
    return real_dec
