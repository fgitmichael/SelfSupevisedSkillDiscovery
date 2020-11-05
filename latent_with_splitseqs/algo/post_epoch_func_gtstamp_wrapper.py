import gtimer as gt

from functools import wraps


def post_epoch_func_wrapper(gt_stamp_name):
    def real_dec(func):
        if gt_stamp_name is not None:
            @wraps(func)
            def wrapper(*args, is_log, **kwargs):
                if is_log:
                    ret = func(*args, **kwargs)
                    gt.stamp(gt_stamp_name)
                else:
                    ret = None
                    gt.stamp(gt_stamp_name)
                return ret
            return wrapper

        else:
            return func
    return real_dec
